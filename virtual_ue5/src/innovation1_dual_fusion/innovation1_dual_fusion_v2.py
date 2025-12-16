"""
创新点1改进版V2: 双路特征融合模型（最小改动版）
Audio + Video 跨模态注意力融合

改进点：
1. 添加FFN层增强特征表达
2. 保留原始特征避免信息丢失
3. 增强融合层（多层MLP）
4. 添加时间建模层
5. 支持注意力权重可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..baseline.base_model import AudioOnlyEncoder, VideoOnlyEncoder, BlendShapeDecoder, HeadPoseDecoder


class CrossModalAttentionV2(nn.Module):
    """改进的跨模态注意力机制 - 支持返回注意力权重"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False
    ) -> tuple:
        """
        Args:
            query: (B, T, dim)
            key_value: (B, T, dim)
            return_attention: 是否返回注意力权重

        Returns:
            output: (B, T, dim)
            attn_weights: (B, num_heads, T, T) or None
        """
        attn_out, attn_weights = self.attention(query, key_value, key_value)
        output = self.norm(query + attn_out)

        if return_attention:
            return output, attn_weights
        return output, None


class FeedForwardNetwork(nn.Module):
    """前馈网络层 - 增强特征表达能力"""

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class DualFusionModelV2(nn.Module):
    """
    双路特征融合模型 V2 - 最小改动版

    改进点：
    1. ✅ 添加FFN层：在跨模态注意力后添加前馈网络
    2. ✅ 保留原始特征：融合时保留audio_feat和video_feat
    3. ✅ 增强融合层：从单层Linear改为多层MLP
    4. ✅ 添加时间建模：融合后添加LSTM层
    5. ✅ 注意力可视化：支持返回注意力权重
    """

    def __init__(
        self,
        audio_dim: int = 80,
        video_dim: int = 478 * 3,
        hidden_dim: int = 256,
        blendshape_dim: int = 52,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_temporal_lstm: bool = True,  # 是否使用时间建模
    ):
        super().__init__()

        self.use_temporal_lstm = use_temporal_lstm

        # 音频和视频编码器
        self.audio_encoder = AudioOnlyEncoder(audio_dim, hidden_dim, num_layers, dropout)
        self.video_encoder = VideoOnlyEncoder(video_dim, hidden_dim, num_layers, dropout)

        encoder_dim = hidden_dim * 2  # LSTM是双向的

        # 跨模态注意力
        self.audio_to_video_attn = CrossModalAttentionV2(encoder_dim, num_heads=8, dropout=dropout)
        self.video_to_audio_attn = CrossModalAttentionV2(encoder_dim, num_heads=8, dropout=dropout)

        # 🆕 改进1: 添加FFN层
        self.audio_ffn = FeedForwardNetwork(encoder_dim, encoder_dim * 2, dropout)
        self.video_ffn = FeedForwardNetwork(encoder_dim, encoder_dim * 2, dropout)

        # 🆕 改进2: 融合时保留原始特征，输入维度变为 encoder_dim * 4
        fusion_input_dim = encoder_dim * 4

        # 🆕 改进3: 增强融合层（多层MLP）
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim),
            nn.LayerNorm(fusion_input_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_input_dim, encoder_dim * 2),
            nn.LayerNorm(encoder_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )

        # 🆕 改进4: 添加时间建模层
        if use_temporal_lstm:
            self.temporal_lstm = nn.LSTM(
                input_size=encoder_dim,
                hidden_size=encoder_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0,
            )
            self.temporal_norm = nn.LayerNorm(encoder_dim)

        # 解码器
        self.blendshape_decoder = BlendShapeDecoder(encoder_dim, hidden_dim, blendshape_dim, dropout)
        self.head_pose_decoder = HeadPoseDecoder(encoder_dim, hidden_dim // 2)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        return_attention: bool = False  # 🆕 改进5: 支持返回注意力权重
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
                'audio_attention': (B, num_heads, T, T) [可选],
                'video_attention': (B, num_heads, T, T) [可选],
            }
        """
        # 1. 编码
        audio_feat = self.audio_encoder(audio)  # (B, T, encoder_dim)
        video_feat = self.video_encoder(video)  # (B, T, encoder_dim)

        # 2. 跨模态注意力 + FFN
        audio_enhanced, audio_attn = self.audio_to_video_attn(
            audio_feat, video_feat, return_attention=return_attention
        )
        audio_enhanced = self.audio_ffn(audio_enhanced)  # 🆕 FFN增强

        video_enhanced, video_attn = self.video_to_audio_attn(
            video_feat, audio_feat, return_attention=return_attention
        )
        video_enhanced = self.video_ffn(video_enhanced)  # 🆕 FFN增强

        # 3. 融合（🆕 保留原始特征）
        fused = torch.cat([
            audio_feat,      # 原始音频特征
            video_feat,      # 原始视频特征
            audio_enhanced,  # 增强后的音频特征
            video_enhanced,  # 增强后的视频特征
        ], dim=-1)  # (B, T, encoder_dim * 4)

        fused = self.fusion(fused)  # (B, T, encoder_dim)

        # 4. 🆕 时间建模
        if self.use_temporal_lstm:
            fused_temporal, _ = self.temporal_lstm(fused)
            fused = self.temporal_norm(fused + fused_temporal)  # 残差连接

        # 5. 解码
        blendshapes = self.blendshape_decoder(fused)
        head_pose = self.head_pose_decoder(fused)

        outputs = {
            'blendshapes': blendshapes,
            'head_pose': head_pose,
        }

        # 🆕 返回注意力权重（用于可视化）
        if return_attention and audio_attn is not None:
            outputs['audio_attention'] = audio_attn
            outputs['video_attention'] = video_attn

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


def create_dual_fusion_model_v2(config: Optional[Dict] = None) -> DualFusionModelV2:
    """创建V2模型的工厂函数"""
    default_config = {
        'audio_dim': 80,
        'video_dim': 478 * 3,
        'hidden_dim': 256,
        'blendshape_dim': 52,
        'num_layers': 2,
        'dropout': 0.1,
        'use_temporal_lstm': True,
    }
    if config:
        default_config.update(config)
    return DualFusionModelV2(**default_config)


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*60)
    print("Testing DualFusionModelV2")
    print("="*60)

    # 创建模型
    model = create_dual_fusion_model_v2().to(device)

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
    if 'audio_attention' in outputs_with_attn:
        print(f"   Audio attention: {outputs_with_attn['audio_attention'].shape}")
        print(f"   Video attention: {outputs_with_attn['video_attention'].shape}")

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
