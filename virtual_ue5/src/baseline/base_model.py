"""
Base模型 - 基础表情驱动模型
仅使用单一输入（音频或视频）驱动MetaHuman BlendShape

这是对比实验的基线模型，用于与创新点模型进行比较。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class AudioOnlyEncoder(nn.Module):
    """仅音频编码器 - 基线方法"""

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) 音频特征
            mask: (B, T) 可选的序列mask，1表示有效，0表示padding
        Returns:
            (B, T, hidden_dim*2) 编码特征
        """
        x = self.input_proj(x)

        # 如果有mask，使用pack_padded_sequence处理变长序列
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            x, _ = self.lstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.lstm(x)

        x = self.layer_norm(x)
        return x


class VideoOnlyEncoder(nn.Module):
    """仅视频编码器 - 基线方法"""

    def __init__(
        self,
        input_dim: int = 478 * 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) 视频特征（关键点）
            mask: (B, T) 可选的序列mask，1表示有效，0表示padding
        Returns:
            (B, T, hidden_dim*2) 编码特征
        """
        B, T, *_ = x.shape
        x = x.view(B, T, -1)
        x = self.input_proj(x)

        # 如果有mask，使用pack_padded_sequence处理变长序列
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            x, _ = self.lstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.lstm(x)

        x = self.layer_norm(x)
        return x


class BlendShapeDecoder(nn.Module):
    """
    BlendShape解码器

    改进：
    - 添加面部区域分组约束
    - 支持表情强度控制
    - 更好的激活函数选择
    """

    # MetaHuman 52个BlendShape的分组
    BROW_RANGE = (0, 8)      # 眉毛：8维
    EYE_RANGE = (8, 22)      # 眼睛：14维
    NOSE_RANGE = (22, 26)    # 鼻子：4维
    MOUTH_RANGE = (26, 46)   # 嘴巴：20维
    CHEEK_RANGE = (46, 52)   # 脸颊：6维

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 52,
        dropout: float = 0.1,
        use_region_constraints: bool = True,
    ):
        super().__init__()

        self.use_region_constraints = use_region_constraints

        # 主解码器
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 区域特定的输出层
        if use_region_constraints:
            # 每个区域有自己的输出层，可以更好地控制表情
            self.brow_head = nn.Linear(hidden_dim // 2, 8)
            self.eye_head = nn.Linear(hidden_dim // 2, 14)
            self.nose_head = nn.Linear(hidden_dim // 2, 4)
            self.mouth_head = nn.Linear(hidden_dim // 2, 20)
            self.cheek_head = nn.Linear(hidden_dim // 2, 6)
        else:
            # 单一输出层
            self.output_layer = nn.Linear(hidden_dim // 2, output_dim)

        # 使用Clamped Sigmoid确保输出在合理范围
        # 而不是简单的Sigmoid，避免梯度消失
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, intensity_scale: float = 1.0) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (B, T, input_dim) 编码特征
            intensity_scale: 全局表情强度缩放因子

        Returns:
            (B, T, 52) BlendShape权重
        """
        # 主解码
        features = self.decoder(x)  # (B, T, hidden_dim//2)

        if self.use_region_constraints:
            # 分区域解码
            outputs = []

            # 眉毛 (0-7)
            brow = self.brow_head(features)
            outputs.append(brow)

            # 眼睛 (8-21)
            eye = self.eye_head(features)
            outputs.append(eye)

            # 鼻子 (22-25)
            nose = self.nose_head(features)
            outputs.append(nose)

            # 嘴巴 (26-45)
            mouth = self.mouth_head(features)
            outputs.append(mouth)

            # 脸颊 (46-51)
            cheek = self.cheek_head(features)
            outputs.append(cheek)

            # 拼接所有区域
            blendshapes = torch.cat(outputs, dim=-1)  # (B, T, 52)
        else:
            # 单一输出层
            blendshapes = self.output_layer(features)

        # 激活函数
        blendshapes = self.output_activation(blendshapes)

        # 应用强度缩放
        if intensity_scale != 1.0:
            blendshapes = blendshapes * intensity_scale
            blendshapes = torch.clamp(blendshapes, 0.0, 1.0)

        return blendshapes

    def get_region_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取各区域的输出（用于可视化或调试）"""
        if not self.use_region_constraints:
            return {'all': self.output_activation(self.output_layer(self.decoder(x)))}

        features = self.decoder(x)
        return {
            'brow': self.output_activation(self.brow_head(features)),
            'eye': self.output_activation(self.eye_head(features)),
            'nose': self.output_activation(self.nose_head(features)),
            'mouth': self.output_activation(self.mouth_head(features)),
            'cheek': self.output_activation(self.cheek_head(features)),
        }


class HeadPoseDecoder(nn.Module):
    """头部姿态解码器 - 输出四元数"""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.decoder(x)
        # 归一化为单位四元数
        q = F.normalize(q, p=2, dim=-1)
        return q


class BaseExpressionModel(nn.Module):
    """
    基础表情驱动模型 (Baseline)

    特点：
    - 单一输入源（音频或视频）
    - 简单的LSTM编码-解码结构
    - 无特征融合
    - 用于对比实验的基线

    输入模式：
    - 'audio': 仅使用音频特征
    - 'video': 仅使用视频特征
    """

    def __init__(
        self,
        mode: str = 'audio',  # 'audio' or 'video'
        audio_dim: int = 80,
        video_dim: int = 478 * 3,
        hidden_dim: int = 256,
        blendshape_dim: int = 52,
        num_layers: int = 2,
        dropout: float = 0.1,
        predict_head_pose: bool = True,
        use_region_constraints: bool = True,  # 新增：是否使用区域约束
    ):
        super().__init__()

        self.mode = mode
        self.predict_head_pose = predict_head_pose

        # 根据模式选择编码器
        if mode == 'audio':
            self.encoder = AudioOnlyEncoder(
                input_dim=audio_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif mode == 'video':
            self.encoder = VideoOnlyEncoder(
                input_dim=video_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        encoder_output_dim = hidden_dim * 2

        # BlendShape解码器 - 支持区域约束
        self.blendshape_decoder = BlendShapeDecoder(
            input_dim=encoder_output_dim,
            hidden_dim=hidden_dim,
            output_dim=blendshape_dim,
            dropout=dropout,
            use_region_constraints=use_region_constraints,
        )

        # 头部姿态解码器
        if predict_head_pose:
            self.head_pose_decoder = HeadPoseDecoder(
                input_dim=encoder_output_dim,
                hidden_dim=hidden_dim // 2,
            )

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            audio: (B, T, audio_dim) 音频特征
            video: (B, T, video_dim) 视频特征
            mask: (B, T) 序列mask，1表示有效，0表示padding

        Returns:
            dict: {
                'blendshapes': (B, T, 52),
                'head_pose': (B, T, 4) [可选]
            }
        """
        # 选择输入
        if self.mode == 'audio':
            if audio is None:
                raise ValueError("Audio input required for audio mode")
            x = audio
        else:
            if video is None:
                raise ValueError("Video input required for video mode")
            x = video

        # 编码
        encoded = self.encoder(x, mask=mask)  # (B, T, hidden*2)

        # 解码BlendShape
        blendshapes = self.blendshape_decoder(encoded)  # (B, T, 52)

        outputs = {'blendshapes': blendshapes}

        # 解码头部姿态
        if self.predict_head_pose:
            head_pose = self.head_pose_decoder(encoded)
            outputs['head_pose'] = head_pose

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失函数

        Args:
            outputs: 模型输出
            targets: 目标值
            weights: 各损失项权重
            mask: (B, T) 序列mask，用于处理变长序列
        """
        if weights is None:
            weights = {
                'blendshape': 1.0,
                'head_pose': 0.5,
                'temporal': 0.1,
                'intensity': 0.01,  # 新增：表情强度正则化
            }

        losses = {}
        device = outputs['blendshapes'].device

        # 创建mask（如果没有提供）
        if mask is None:
            mask = torch.ones_like(outputs['blendshapes'][..., 0])  # (B, T)

        # BlendShape损失（带mask）
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
        blendshape_loss = F.mse_loss(
            outputs['blendshapes'] * mask_expanded,
            targets['blendshapes'] * mask_expanded,
            reduction='sum'
        ) / mask.sum()
        losses['blendshape_loss'] = blendshape_loss

        # 头部姿态损失（四元数距离）
        if 'head_pose' in outputs and 'head_pose' in targets:
            # 四元数可以用点积衡量相似度
            pred_quat = outputs['head_pose']  # (B, T, 4)
            target_quat = targets['head_pose']  # (B, T, 4)

            # 计算四元数之间的角度差异
            # 使用安全的替代公式避免 torch.acos() 的梯度爆炸
            quat_dot = (pred_quat * target_quat).sum(dim=-1).abs()  # (B, T)
            quat_dot = torch.clamp(quat_dot, 0.0, 1.0)  # 确保在有效范围内
            # 使用 2 * (1 - quat_dot) 作为角度差异的近似，避免 acos 的数值不稳定
            # 当 quat_dot=1 时差异为0，quat_dot=0 时差异为2（相当于π/2的缩放版本）
            angular_diff = 2.0 * (1.0 - quat_dot)

            # 应用mask并计算损失
            head_pose_loss = (angular_diff * mask).sum() / mask.sum()
            losses['head_pose_loss'] = head_pose_loss

        # 时间一致性损失 - 修正：对比预测和目标的时间变化
        if outputs['blendshapes'].size(1) > 1:
            pred_diff = outputs['blendshapes'][:, 1:] - outputs['blendshapes'][:, :-1]
            target_diff = targets['blendshapes'][:, 1:] - targets['blendshapes'][:, :-1]

            # 对差值也应用mask（转换为bool类型）
            mask_diff = mask[:, 1:].bool() & mask[:, :-1].bool()  # 两个时间步都需要有效
            mask_diff = mask_diff.float()  # 转回float用于计算
            temporal_loss = F.mse_loss(
                pred_diff * mask_diff.unsqueeze(-1),
                target_diff * mask_diff.unsqueeze(-1),
                reduction='sum'
            ) / mask_diff.sum().clamp(min=1)
        else:
            temporal_loss = torch.tensor(0.0, device=device)
        losses['temporal_loss'] = temporal_loss

        # 表情强度正则化（防止过度夸张）
        intensity_loss = (outputs['blendshapes'] ** 2 * mask_expanded).sum() / mask.sum()
        losses['intensity_loss'] = intensity_loss

        # 总损失 - 修正tensor类型问题
        total_loss = (
            weights['blendshape'] * blendshape_loss +
            weights.get('head_pose', 0.5) * losses.get('head_pose_loss', torch.tensor(0.0, device=device)) +
            weights['temporal'] * temporal_loss +
            weights.get('intensity', 0.01) * intensity_loss
        )
        losses['total_loss'] = total_loss

        return losses


def create_base_model(
    mode: str = 'audio',
    config: Optional[Dict] = None,
) -> BaseExpressionModel:
    """创建基础模型的工厂函数"""

    default_config = {
        'audio_dim': 80,
        'video_dim': 478 * 3,
        'hidden_dim': 256,
        'blendshape_dim': 52,
        'num_layers': 2,
        'dropout': 0.1,
        'predict_head_pose': True,
    }

    if config:
        default_config.update(config)

    return BaseExpressionModel(mode=mode, **default_config)


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 测试音频模式
    print("\n--- Testing Audio Mode ---")
    model_audio = create_base_model(mode='audio').to(device)
    audio_input = torch.randn(4, 100, 80).to(device)

    outputs = model_audio(audio=audio_input)
    print(f"BlendShapes shape: {outputs['blendshapes'].shape}")
    print(f"Head pose shape: {outputs['head_pose'].shape}")

    # 测试视频模式
    print("\n--- Testing Video Mode ---")
    model_video = create_base_model(mode='video').to(device)
    video_input = torch.randn(4, 100, 478 * 3).to(device)

    outputs = model_video(video=video_input)
    print(f"BlendShapes shape: {outputs['blendshapes'].shape}")
    print(f"Head pose shape: {outputs['head_pose'].shape}")

    # 测试损失计算
    print("\n--- Testing Loss Computation ---")
    targets = {
        'blendshapes': torch.rand(4, 100, 52).to(device),
        'head_pose': F.normalize(torch.randn(4, 100, 4), dim=-1).to(device),
    }

    losses = model_audio.compute_loss(
        model_audio(audio=audio_input),
        targets
    )
    for name, value in losses.items():
        print(f"{name}: {value.item():.4f}")

    print("\n--- Model Parameters ---")
    print(f"Audio model params: {sum(p.numel() for p in model_audio.parameters()):,}")
    print(f"Video model params: {sum(p.numel() for p in model_video.parameters()):,}")
