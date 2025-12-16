"""
创新点1改进版V3: 双路特征融合模型（完整改进版）
Audio + Video 跨模态注意力融合

改进点：
1. 多层跨模态交互
2. 自适应门控融合机制
3. 增强的时间建模
4. 注意力权重可视化
5. 更强的特征表达能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

from ..baseline.base_model import AudioOnlyEncoder, VideoOnlyEncoder, BlendShapeDecoder, HeadPoseDecoder


class CrossModalBlock(nn.Module):
    """跨模态交互块 - 包含双向注意力和FFN"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # 音频到视频的注意力
        self.audio_to_video_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.audio_norm1 = nn.LayerNorm(dim)

        # 视频到音频的注意力
        self.video_to_audio_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.video_norm1 = nn.LayerNorm(dim)

        # FFN for audio
        self.audio_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.audio_norm2 = nn.LayerNorm(dim)

        # FFN for video
        self.video_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.video_norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        return_attention: bool = False
    ) -> tuple:
        """
        Args:
            audio_feat: (B, T, dim)
            video_feat: (B, T, dim)
            return_attention: 是否返回注意力权重

        Returns:
            audio_out: (B, T, dim)
            video_out: (B, T, dim)
            attn_weights: dict or None
        """
        # 跨模态注意力
        audio_attn_out, audio_attn_weights = self.audio_to_video_attn(
            audio_feat, video_feat, video_feat
        )
        audio_feat = self.audio_norm1(audio_feat + audio_attn_out)

        video_attn_out, video_attn_weights = self.video_to_audio_attn(
            video_feat, audio_feat, audio_feat
        )
        video_feat = self.video_norm1(video_feat + video_attn_out)

        # FFN
        audio_feat = self.audio_norm2(audio_feat + self.audio_ffn(audio_feat))
        video_feat = self.video_norm2(video_feat + self.video_ffn(video_feat))

        attn_weights = None
        if return_attention:
            attn_weights = {
                'audio_to_video': audio_attn_weights,
                'video_to_audio': video_attn_weights,
            }

        return audio_feat, video_feat, attn_weights


class AdaptiveFusionGate(nn.Module):
    """自适应融合门控机制 - 动态学习融合权重"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1),  # 权重归一化
        )

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_feat: (B, T, dim)
            video_feat: (B, T, dim)

        Returns:
            fused: (B, T, dim)
        """
        # 拼接特征
        concat = torch.cat([audio_feat, video_feat], dim=-1)

        # 计算门控权重
        weights = self.gate_net(concat)  # (B, T, 2)

        # 加权融合
        audio_weight = weights[..., 0:1]  # (B, T, 1)
        video_weight = weights[..., 1:2]  # (B, T, 1)

        fused = audio_weight * audio_feat + video_weight * video_feat

        return fused


class DualFusionModelV3(nn.Module):
    """
    双路特征融合模型 V3 - 完整改进版

    改进点：
    1. ✅ 多层跨模态交互：堆叠多个CrossModalBlock
    2. ✅ 自适应融合：使用门控机制动态学习融合权重
    3. ✅ 增强时间建模：双层LSTM + Temporal Attention
    4. ✅ 注意力可视化：保存所有层的注意力权重
    5. ✅ 更强表达能力：更深的网络结构
    """

    def __init__(
        self,
        audio_dim: int = 80,
        video_dim: int = 478 * 3,
        hidden_dim: int = 256,
        blendshape_dim: int = 52,
        num_layers: int = 2,
        num_cross_modal_layers: int = 2,  # 跨模态交互层数
        dropout: float = 0.1,
        use_adaptive_fusion: bool = True,  # 是否使用自适应融合
    ):
        super().__init__()

        self.num_cross_modal_layers = num_cross_modal_layers
        self.use_adaptive_fusion = use_adaptive_fusion

        # 音频和视频编码器
        self.audio_encoder = AudioOnlyEncoder(audio_dim, hidden_dim, num_layers, dropout)
        self.video_encoder = VideoOnlyEncoder(video_dim, hidden_dim, num_layers, dropout)

        encoder_dim = hidden_dim * 2  # LSTM是双向的

        # 🆕 多层跨模态交互
        self.cross_modal_layers = nn.ModuleList([
            CrossModalBlock(encoder_dim, num_heads=8, dropout=dropout)
            for _ in range(num_cross_modal_layers)
        ])

        # 🆕 自适应融合门控
        if use_adaptive_fusion:
            self.fusion_gate = AdaptiveFusionGate(encoder_dim, dropout)
            fusion_dim = encoder_dim
        else:
            # 传统拼接融合
            fusion_dim = encoder_dim * 2

        # 融合后的处理
        self.post_fusion = nn.Sequential(
            nn.Linear(fusion_dim, encoder_dim * 2),
            nn.LayerNorm(encoder_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )

        # 🆕 增强的时间建模
        self.temporal_lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=encoder_dim // 2,
            num_layers=2,  # 双层LSTM
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.temporal_norm = nn.LayerNorm(encoder_dim)

        # 时间注意力（自注意力）
        self.temporal_attention = nn.MultiheadAttention(
            encoder_dim, num_heads=8, batch_first=True, dropout=dropout
        )
        self.temporal_attn_norm = nn.LayerNorm(encoder_dim)

        # 解码器
        self.blendshape_decoder = BlendShapeDecoder(encoder_dim, hidden_dim, blendshape_dim, dropout)
        self.head_pose_decoder = HeadPoseDecoder(encoder_dim, hidden_dim // 2)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            audio: (B, T, 80) 音频特征
            video: (B, T, 1434) 视频特征
            return_attention: 是否返回注意力权重用于可视化

        Returns:
            dict: {
                'blendshapes': (B, T, 52),
                'head_pose': (B, T, 4),
                'cross_modal_attention': List[dict] [可选],
                'temporal_attention': (B, num_heads, T, T) [可选],
                'fusion_weights': (B, T, 2) [可选, 仅adaptive fusion],
            }
        """
        # 1. 编码
        audio_feat = self.audio_encoder(audio)  # (B, T, encoder_dim)
        video_feat = self.video_encoder(video)  # (B, T, encoder_dim)

        # 2. 🆕 多层跨模态交互
        cross_modal_attentions = []
        for i, cross_modal_layer in enumerate(self.cross_modal_layers):
            audio_feat, video_feat, attn_weights = cross_modal_layer(
                audio_feat, video_feat, return_attention=return_attention
            )
            if return_attention and attn_weights is not None:
                cross_modal_attentions.append(attn_weights)

        # 3. 🆕 自适应融合
        if self.use_adaptive_fusion:
            fused = self.fusion_gate(audio_feat, video_feat)
            # 保存融合权重用于分析
            if return_attention:
                # 重新计算门控权重用于返回
                concat = torch.cat([audio_feat, video_feat], dim=-1)
                fusion_weights = self.fusion_gate.gate_net(concat)
            else:
                fusion_weights = None
        else:
            fused = torch.cat([audio_feat, video_feat], dim=-1)
            fusion_weights = None

        # 4. 融合后处理
        fused = self.post_fusion(fused)  # (B, T, encoder_dim)

        # 5. 🆕 增强的时间建模
        # 5.1 双层LSTM
        fused_temporal, _ = self.temporal_lstm(fused)
        fused = self.temporal_norm(fused + fused_temporal)

        # 5.2 时间自注意力
        temporal_attn_out, temporal_attn_weights = self.temporal_attention(
            fused, fused, fused
        )
        fused = self.temporal_attn_norm(fused + temporal_attn_out)

        # 6. 解码
        blendshapes = self.blendshape_decoder(fused)
        head_pose = self.head_pose_decoder(fused)

        outputs = {
            'blendshapes': blendshapes,
            'head_pose': head_pose,
        }

        # 🆕 返回注意力权重和融合权重（用于可视化和分析）
        if return_attention:
            if len(cross_modal_attentions) > 0:
                outputs['cross_modal_attention'] = cross_modal_attentions
            outputs['temporal_attention'] = temporal_attn_weights
            if fusion_weights is not None:
                outputs['fusion_weights'] = fusion_weights

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失函数

        Args:
            outputs: 模型输出
            targets: 目标值
        """
        losses = {}

        # BlendShape损失
        losses['blendshape_loss'] = F.mse_loss(outputs['blendshapes'], targets['blendshapes'])

        # 头部姿态损失
        losses['head_pose_loss'] = F.mse_loss(outputs['head_pose'], targets['head_pose'])

        # 时间平滑度损失
        pred_diff = outputs['blendshapes'][:, 1:] - outputs['blendshapes'][:, :-1]
        losses['temporal_loss'] = (pred_diff ** 2).mean()

        # 总损失
        losses['total_loss'] = (
            losses['blendshape_loss'] +
            0.5 * losses['head_pose_loss'] +
            0.1 * losses['temporal_loss']
        )

        return losses


def create_dual_fusion_model_v3(config: Optional[Dict] = None) -> DualFusionModelV3:
    """创建V3模型的工厂函数"""
    default_config = {
        'audio_dim': 80,
        'video_dim': 478 * 3,
        'hidden_dim': 256,
        'blendshape_dim': 52,
        'num_layers': 2,
        'num_cross_modal_layers': 2,
        'dropout': 0.1,
        'use_adaptive_fusion': True,
    }
    if config:
        default_config.update(config)
    return DualFusionModelV3(**default_config)


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*60)
    print("Testing DualFusionModelV3")
    print("="*60)

    # 创建模型
    model = create_dual_fusion_model_v3().to(device)

    # 生成测试数据
    batch_size, seq_len = 4, 100
    audio_input = torch.randn(batch_size, seq_len, 80).to(device)
    video_input = torch.randn(batch_size, seq_len, 478 * 3).to(device)

    # 前向传播
    print("\n1. Forward pass without attention:")
    outputs = model(audio=audio_input, video=video_input)
    print(f"   BlendShapes: {outputs['blendshapes'].shape}")
    print(f"   Head pose: {outputs['head_pose'].shape}")

    # 前向传播（带注意力权重）
    print("\n2. Forward pass with attention:")
    outputs_with_attn = model(audio=audio_input, video=video_input, return_attention=True)
    print(f"   BlendShapes: {outputs_with_attn['blendshapes'].shape}")
    print(f"   Head pose: {outputs_with_attn['head_pose'].shape}")
    if 'cross_modal_attention' in outputs_with_attn:
        print(f"   Cross-modal attention layers: {len(outputs_with_attn['cross_modal_attention'])}")
    if 'temporal_attention' in outputs_with_attn:
        print(f"   Temporal attention: {outputs_with_attn['temporal_attention'].shape}")
    if 'fusion_weights' in outputs_with_attn:
        print(f"   Fusion weights: {outputs_with_attn['fusion_weights'].shape}")
        # 打印平均融合权重
        avg_weights = outputs_with_attn['fusion_weights'].mean(dim=[0, 1])
        print(f"   Avg audio weight: {avg_weights[0]:.4f}, Avg video weight: {avg_weights[1]:.4f}")

    # 损失计算
    print("\n3. Loss computation:")
    targets = {
        'blendshapes': torch.rand(batch_size, seq_len, 52).to(device),
        'head_pose': F.normalize(torch.randn(batch_size, seq_len, 4), dim=-1).to(device),
    }
    losses = model.compute_loss(outputs, targets)
    for name, value in losses.items():
        print(f"   {name}: {value.item():.6f}")

    # 模型参数
    print("\n4. Model info:")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params:,}")
    print(f"   Size: {num_params / 1e6:.2f}M")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
