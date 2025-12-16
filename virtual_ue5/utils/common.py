"""
通用工具函数
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_synthetic_data(
    batch_size: int = 8,
    seq_len: int = 100,
    audio_dim: int = 80,
    video_dim: int = 478 * 3,
    blendshape_dim: int = 52,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    生成合成训练数据用于测试

    Returns:
        dict: 包含audio, video, blendshapes的字典
    """
    if device is None:
        device = get_device()

    # 生成音频特征 (Mel频谱)
    audio = torch.randn(batch_size, seq_len, audio_dim, device=device)

    # 生成视频特征 (面部关键点)
    video = torch.randn(batch_size, seq_len, video_dim, device=device) * 0.1 + 0.5

    # 生成BlendShape目标 (0-1范围)
    blendshapes = torch.sigmoid(torch.randn(batch_size, seq_len, blendshape_dim, device=device))

    # 生成头部姿态 (四元数)
    head_pose = torch.randn(batch_size, seq_len, 4, device=device)
    head_pose = head_pose / head_pose.norm(dim=-1, keepdim=True)

    return {
        'audio': audio,
        'video': video,
        'blendshapes': blendshapes,
        'head_pose': head_pose,
    }


def blendshape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算BlendShape损失

    Args:
        pred: (B, T, 52) 预测的BlendShape
        target: (B, T, 52) 目标BlendShape
        weights: (52,) 各BlendShape的权重
    """
    mse = (pred - target) ** 2

    if weights is not None:
        mse = mse * weights.unsqueeze(0).unsqueeze(0)

    return mse.mean()


def temporal_consistency_loss(
    pred: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """
    时间一致性损失 - 确保预测序列平滑

    Args:
        pred: (B, T, D) 预测序列
        order: 差分阶数 (1=速度, 2=加速度)
    """
    diff = pred[:, 1:] - pred[:, :-1]  # 一阶差分

    if order >= 2:
        diff = diff[:, 1:] - diff[:, :-1]  # 二阶差分

    return (diff ** 2).mean()


def lip_sync_loss(
    pred_blendshapes: torch.Tensor,
    audio_features: torch.Tensor,
) -> torch.Tensor:
    """
    口型同步损失 - 确保口型与音频匹配

    Args:
        pred_blendshapes: (B, T, 52) 预测的BlendShape
        audio_features: (B, T, D) 音频特征
    """
    # 提取口部相关的BlendShape (索引17-41大致是口部)
    mouth_bs = pred_blendshapes[:, :, 17:42]

    # 计算音频能量
    audio_energy = audio_features.pow(2).mean(dim=-1, keepdim=True)

    # 计算口部运动量
    mouth_movement = (mouth_bs[:, 1:] - mouth_bs[:, :-1]).pow(2).mean(dim=-1, keepdim=True)

    # 确保维度匹配
    audio_energy = audio_energy[:, 1:]

    # 口型运动应该与音频能量正相关
    correlation = -torch.mean(mouth_movement * audio_energy)

    return correlation


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转矩阵

    Args:
        q: (..., 4) 四元数 [w, x, y, z]
    Returns:
        (..., 3, 3) 旋转矩阵
    """
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
        2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
        2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y,
    ], dim=-1).view(*q.shape[:-1], 3, 3)

    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为四元数

    Args:
        R: (..., 3, 3) 旋转矩阵
    Returns:
        (..., 4) 四元数 [w, x, y, z]
    """
    batch_shape = R.shape[:-2]
    R = R.view(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    s = torch.sqrt(trace + 1.0) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s

    return q.view(*batch_shape, 4)
