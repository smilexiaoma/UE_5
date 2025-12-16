"""
源代码模块
包含 baseline 和所有创新点的模型实现
"""

# 导入 baseline 模型
from .baseline.base_model import BaseExpressionModel, create_base_model

# 导入创新点模型
from .innovation1_dual_fusion.innovation1_dual_fusion import DualFusionModel, create_dual_fusion_model
from .innovation2_diffusion.innovation2_diffusion import DiffusionExpressionModel, create_diffusion_model
from .innovation3_e2e_loop.innovation3_e2e_loop import E2ELoopModel, create_e2e_loop_model

__all__ = [
    'BaseExpressionModel',
    'create_base_model',
    'DualFusionModel',
    'create_dual_fusion_model',
    'DiffusionExpressionModel',
    'create_diffusion_model',
    'E2ELoopModel',
    'create_e2e_loop_model',
]
