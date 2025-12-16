"""
创新点3: 端到端闭环系统
渲染状态反馈 + 实时误差校正
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..baseline.base_model import AudioOnlyEncoder, VideoOnlyEncoder, BlendShapeDecoder, HeadPoseDecoder


class FeedbackModule(nn.Module):
    """反馈模块 - 模拟渲染状态反馈"""

    def __init__(self, blendshape_dim: int = 52, hidden_dim: int = 128):
        super().__init__()

        self.feedback_encoder = nn.Sequential(
            nn.Linear(blendshape_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, blendshapes: torch.Tensor) -> torch.Tensor:
        """编码反馈信号"""
        return self.feedback_encoder(blendshapes)


class ErrorCorrectionModule(nn.Module):
    """误差校正模块"""

    def __init__(self, feature_dim: int, feedback_dim: int, output_dim: int = 52):
        super().__init__()

        self.correction = nn.Sequential(
            nn.Linear(feature_dim + feedback_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim),
            nn.Tanh(),  # 校正量在 [-1, 1]
        )

    def forward(self, features: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        """计算校正量"""
        combined = torch.cat([features, feedback], dim=-1)
        correction = self.correction(combined)
        return correction * 0.1  # 限制校正幅度


class E2ELoopModel(nn.Module):
    """端到端闭环模型"""

    def __init__(
        self,
        audio_dim: int = 80,
        video_dim: int = 478 * 3,
        hidden_dim: int = 256,
        blendshape_dim: int = 52,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_iterations: int = 2,
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.blendshape_dim = blendshape_dim

        # 编码器
        self.audio_encoder = AudioOnlyEncoder(audio_dim, hidden_dim, num_layers, dropout)
        self.video_encoder = VideoOnlyEncoder(video_dim, hidden_dim, num_layers, dropout)

        encoder_dim = hidden_dim * 4

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.LayerNorm(encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 初始预测器
        self.initial_predictor = BlendShapeDecoder(
            encoder_dim // 2, hidden_dim, blendshape_dim, dropout
        )

        # 反馈和校正模块
        self.feedback_module = FeedbackModule(blendshape_dim, hidden_dim)
        self.correction_module = ErrorCorrectionModule(
            encoder_dim // 2, hidden_dim, blendshape_dim
        )

        # 头部姿态预测器
        self.head_pose_decoder = HeadPoseDecoder(encoder_dim // 2, hidden_dim // 2)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        num_iterations: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 迭代优化

        Args:
            audio: (B, T, audio_dim)
            video: (B, T, video_dim)
            num_iterations: 迭代次数 (默认使用初始化时的值)
        """
        if num_iterations is None:
            num_iterations = self.num_iterations

        # 编码输入
        audio_feat = self.audio_encoder(audio)
        video_feat = self.video_encoder(video)
        features = torch.cat([audio_feat, video_feat], dim=-1)
        features = self.fusion(features)

        # 初始预测
        blendshapes = self.initial_predictor(features)

        # 迭代优化
        for i in range(num_iterations):
            # 获取反馈
            feedback = self.feedback_module(blendshapes)

            # 计算校正
            correction = self.correction_module(features, feedback)

            # 应用校正
            blendshapes = blendshapes + correction
            blendshapes = torch.clamp(blendshapes, 0, 1)

        # 预测头部姿态
        head_pose = self.head_pose_decoder(features)

        return {
            'blendshapes': blendshapes,
            'head_pose': head_pose,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        losses = {}

        # BlendShape 损失
        losses['blendshape_loss'] = F.mse_loss(
            outputs['blendshapes'],
            targets['blendshapes']
        )

        # 头部姿态损失
        losses['head_pose_loss'] = F.mse_loss(
            outputs['head_pose'],
            targets['head_pose']
        )

        # 时序平滑损失 (减少抖动)
        pred_diff = outputs['blendshapes'][:, 1:] - outputs['blendshapes'][:, :-1]
        target_diff = targets['blendshapes'][:, 1:] - targets['blendshapes'][:, :-1]
        losses['temporal_loss'] = F.mse_loss(pred_diff, target_diff)

        # 总损失
        losses['total_loss'] = (
            losses['blendshape_loss'] +
            0.5 * losses['head_pose_loss'] +
            0.2 * losses['temporal_loss']
        )

        return losses


def create_e2e_loop_model(config: Optional[Dict] = None) -> E2ELoopModel:
    """创建端到端闭环模型"""
    default_config = {
        'audio_dim': 80,
        'video_dim': 478 * 3,
        'hidden_dim': 256,
        'blendshape_dim': 52,
        'num_layers': 2,
        'dropout': 0.1,
        'num_iterations': 2,
    }
    if config:
        default_config.update(config)
    return E2ELoopModel(**default_config)
