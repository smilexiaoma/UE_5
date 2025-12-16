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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) 音频特征
        Returns:
            (B, T, hidden_dim*2) 编码特征
        """
        x = self.input_proj(x)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) 视频特征（关键点）
        Returns:
            (B, T, hidden_dim*2) 编码特征
        """
        B, T, *_ = x.shape
        x = x.view(B, T, -1)
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return x


class BlendShapeDecoder(nn.Module):
    """BlendShape解码器"""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 52,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),  # BlendShape权重在0-1范围
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


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

        # BlendShape解码器
        self.blendshape_decoder = BlendShapeDecoder(
            input_dim=encoder_output_dim,
            hidden_dim=hidden_dim,
            output_dim=blendshape_dim,
            dropout=dropout,
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
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            audio: (B, T, audio_dim) 音频特征
            video: (B, T, video_dim) 视频特征

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
        encoded = self.encoder(x)  # (B, T, hidden*2)

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
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失函数

        Args:
            outputs: 模型输出
            targets: 目标值
            weights: 各损失项权重
        """
        if weights is None:
            weights = {
                'blendshape': 1.0,
                'head_pose': 0.5,
                'temporal': 0.1,
            }

        losses = {}

        # BlendShape损失
        blendshape_loss = F.mse_loss(
            outputs['blendshapes'],
            targets['blendshapes']
        )
        losses['blendshape_loss'] = blendshape_loss

        # 头部姿态损失
        if 'head_pose' in outputs and 'head_pose' in targets:
            head_pose_loss = F.mse_loss(
                outputs['head_pose'],
                targets['head_pose']
            )
            losses['head_pose_loss'] = head_pose_loss

        # 时间一致性损失
        pred_diff = outputs['blendshapes'][:, 1:] - outputs['blendshapes'][:, :-1]
        temporal_loss = (pred_diff ** 2).mean()
        losses['temporal_loss'] = temporal_loss

        # 总损失
        total_loss = (
            weights['blendshape'] * blendshape_loss +
            weights.get('head_pose', 0.5) * losses.get('head_pose_loss', 0) +
            weights['temporal'] * temporal_loss
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
