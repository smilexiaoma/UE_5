"""
创新点1: 双路特征融合模型
Audio + Video 跨模态注意力融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .base_model import AudioOnlyEncoder, VideoOnlyEncoder, BlendShapeDecoder, HeadPoseDecoder


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(query, key_value, key_value)
        return self.norm(query + attn_out)


class DualFusionModel(nn.Module):
    """双路特征融合模型"""

    def __init__(
        self,
        audio_dim: int = 80,
        video_dim: int = 478 * 3,
        hidden_dim: int = 256,
        blendshape_dim: int = 52,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 音频和视频编码器
        self.audio_encoder = AudioOnlyEncoder(audio_dim, hidden_dim, num_layers, dropout)
        self.video_encoder = VideoOnlyEncoder(video_dim, hidden_dim, num_layers, dropout)

        encoder_dim = hidden_dim * 2

        # 跨模态注意力
        self.audio_to_video_attn = CrossModalAttention(encoder_dim)
        self.video_to_audio_attn = CrossModalAttention(encoder_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 解码器
        self.blendshape_decoder = BlendShapeDecoder(encoder_dim, hidden_dim, blendshape_dim, dropout)
        self.head_pose_decoder = HeadPoseDecoder(encoder_dim, hidden_dim // 2)

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码
        audio_feat = self.audio_encoder(audio)
        video_feat = self.video_encoder(video)

        # 跨模态注意力
        audio_enhanced = self.audio_to_video_attn(audio_feat, video_feat)
        video_enhanced = self.video_to_audio_attn(video_feat, audio_feat)

        # 融合
        fused = torch.cat([audio_enhanced, video_enhanced], dim=-1)
        fused = self.fusion(fused)

        # 解码
        blendshapes = self.blendshape_decoder(fused)
        head_pose = self.head_pose_decoder(fused)

        return {'blendshapes': blendshapes, 'head_pose': head_pose}

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        losses['blendshape_loss'] = F.mse_loss(outputs['blendshapes'], targets['blendshapes'])
        losses['head_pose_loss'] = F.mse_loss(outputs['head_pose'], targets['head_pose'])

        pred_diff = outputs['blendshapes'][:, 1:] - outputs['blendshapes'][:, :-1]
        losses['temporal_loss'] = (pred_diff ** 2).mean()

        losses['total_loss'] = losses['blendshape_loss'] + 0.5 * losses['head_pose_loss'] + 0.1 * losses['temporal_loss']
        return losses


def create_dual_fusion_model(config: Optional[Dict] = None) -> DualFusionModel:
    default_config = {
        'audio_dim': 80,
        'video_dim': 478 * 3,
        'hidden_dim': 256,
        'blendshape_dim': 52,
        'num_layers': 2,
        'dropout': 0.1,
    }
    if config:
        default_config.update(config)
    return DualFusionModel(**default_config)
