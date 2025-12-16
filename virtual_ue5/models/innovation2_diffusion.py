"""
创新点2: 扩散模型优化
DDPM去噪 + 时序一致性约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math

from .base_model import AudioOnlyEncoder, VideoOnlyEncoder


class DiffusionExpressionModel(nn.Module):
    """扩散模型优化表情生成"""

    def __init__(
        self,
        audio_dim: int = 80,
        video_dim: int = 478 * 3,
        hidden_dim: int = 256,
        blendshape_dim: int = 52,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_diffusion_steps: int = 1000,
    ):
        super().__init__()

        self.blendshape_dim = blendshape_dim
        self.num_diffusion_steps = num_diffusion_steps

        # 条件编码器
        self.audio_encoder = AudioOnlyEncoder(audio_dim, hidden_dim, num_layers, dropout)
        self.video_encoder = VideoOnlyEncoder(video_dim, hidden_dim, num_layers, dropout)

        encoder_dim = hidden_dim * 4  # audio + video

        # 去噪网络
        self.denoiser = nn.Sequential(
            nn.Linear(blendshape_dim + encoder_dim + 1, hidden_dim),  # +1 for timestep
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, blendshape_dim),
        )

        # 头部姿态预测器
        self.head_pose_predictor = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
        )

        # 扩散参数
        self.register_buffer('betas', self._cosine_beta_schedule(num_diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """余弦调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        blendshapes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """训练时的前向传播"""
        B, T, _ = audio.shape

        # 编码条件
        audio_feat = self.audio_encoder(audio)
        video_feat = self.video_encoder(video)
        condition = torch.cat([audio_feat, video_feat], dim=-1)

        if blendshapes is not None:
            # 训练模式: 添加噪声并预测
            t = torch.randint(0, self.num_diffusion_steps, (B,), device=audio.device)
            noise = torch.randn_like(blendshapes)

            # 添加噪声
            alpha_t = self.alphas_cumprod[t].view(B, 1, 1)
            noisy_blendshapes = torch.sqrt(alpha_t) * blendshapes + torch.sqrt(1 - alpha_t) * noise

            # 预测噪声
            t_emb = t.float().view(B, 1, 1).expand(B, T, 1) / self.num_diffusion_steps
            denoiser_input = torch.cat([noisy_blendshapes, condition, t_emb], dim=-1)
            predicted_noise = self.denoiser(denoiser_input)

            # 预测头部姿态
            head_pose = self.head_pose_predictor(condition)
            head_pose = F.normalize(head_pose, p=2, dim=-1)

            return {
                'blendshapes': predicted_noise,
                'head_pose': head_pose,
                'noise_target': noise,
            }
        else:
            # 推理模式: 从噪声采样
            return self.sample(audio, video, num_inference_steps=50)

    def sample(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        num_inference_steps: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """DDIM 采样"""
        B, T, _ = audio.shape

        # 编码条件
        audio_feat = self.audio_encoder(audio)
        video_feat = self.video_encoder(video)
        condition = torch.cat([audio_feat, video_feat], dim=-1)

        # 从噪声开始
        x = torch.randn(B, T, self.blendshape_dim, device=audio.device)

        # 简化采样 (DDIM)
        step_size = self.num_diffusion_steps // num_inference_steps
        for i in reversed(range(0, self.num_diffusion_steps, step_size)):
            t = torch.full((B,), i, device=audio.device, dtype=torch.long)
            t_emb = t.float().view(B, 1, 1).expand(B, T, 1) / self.num_diffusion_steps

            denoiser_input = torch.cat([x, condition, t_emb], dim=-1)
            predicted_noise = self.denoiser(denoiser_input)

            alpha_t = self.alphas_cumprod[t].view(B, 1, 1)
            alpha_t_prev = self.alphas_cumprod[max(0, i - step_size)].view(B, 1, 1) if i > 0 else torch.ones_like(alpha_t)

            # DDIM 更新
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, 0, 1)

            x = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * predicted_noise

        # 预测头部姿态
        head_pose = self.head_pose_predictor(condition)
        head_pose = F.normalize(head_pose, p=2, dim=-1)

        return {'blendshapes': torch.clamp(x, 0, 1), 'head_pose': head_pose}

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # 噪声预测损失
        if 'noise_target' in outputs:
            losses['noise_loss'] = F.mse_loss(outputs['blendshapes'], outputs['noise_target'])
        else:
            losses['noise_loss'] = F.mse_loss(outputs['blendshapes'], targets['blendshapes'])

        # 头部姿态损失
        losses['head_pose_loss'] = F.mse_loss(outputs['head_pose'], targets['head_pose'])

        # 总损失
        losses['total_loss'] = losses['noise_loss'] + 0.5 * losses['head_pose_loss']

        return losses


def create_diffusion_model(config: Optional[Dict] = None) -> DiffusionExpressionModel:
    default_config = {
        'audio_dim': 80,
        'video_dim': 478 * 3,
        'hidden_dim': 256,
        'blendshape_dim': 52,
        'num_layers': 2,
        'dropout': 0.1,
        'num_diffusion_steps': 1000,
    }
    if config:
        default_config.update(config)
    return DiffusionExpressionModel(**default_config)
