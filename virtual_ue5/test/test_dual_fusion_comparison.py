"""
对比测试脚本：DualFusionModel 三个版本的性能对比

对比版本：
- V1: 原始版本 (innovation1_dual_fusion.py)
- V2: 最小改动版 (innovation1_dual_fusion_v2.py)
- V3: 完整改进版 (innovation1_dual_fusion_v3.py)

对比维度：
1. 模型参数量
2. 前向传播速度
3. 输出形状和正确性
4. 特性对比
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from typing import Dict, List
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.innovation1_dual_fusion.innovation1_dual_fusion import create_dual_fusion_model
from src.innovation1_dual_fusion.innovation1_dual_fusion_v2 import create_dual_fusion_model_v2
from src.innovation1_dual_fusion.innovation1_dual_fusion_v3 import create_dual_fusion_model_v3


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())


def measure_inference_time(
    model: torch.nn.Module,
    audio: torch.Tensor,
    video: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10,
    device: torch.device = None
) -> Dict[str, float]:
    """测量推理时间"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(audio=audio, video=video)

    # 同步
    if device and device.type == 'cuda':
        torch.cuda.synchronize()

    # 计时
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(audio=audio, video=video)

            if device and device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
    }


def test_model_features(model_name: str, model: torch.nn.Module, audio: torch.Tensor, video: torch.Tensor):
    """测试模型的特性"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)

    model.eval()

    # 1. 参数量
    num_params = count_parameters(model)
    print(f"\n1. Model Parameters:")
    print(f"   Total: {num_params:,}")
    print(f"   Size: {num_params / 1e6:.2f}M")

    # 2. 前向传播
    print(f"\n2. Forward Pass:")
    with torch.no_grad():
        try:
            outputs = model(audio=audio, video=video)
            print(f"   ✓ Basic forward pass successful")
            print(f"   BlendShapes shape: {outputs['blendshapes'].shape}")
            print(f"   Head pose shape: {outputs['head_pose'].shape}")
        except Exception as e:
            print(f"   ✗ Basic forward pass failed: {e}")
            return None

    # 3. 测试return_attention功能（V2和V3支持）
    print(f"\n3. Attention Weights:")
    if model_name in ['V2', 'V3']:
        try:
            with torch.no_grad():
                outputs_with_attn = model(audio=audio, video=video, return_attention=True)

            if 'audio_attention' in outputs_with_attn:
                print(f"   ✓ Audio attention: {outputs_with_attn['audio_attention'].shape}")
            if 'video_attention' in outputs_with_attn:
                print(f"   ✓ Video attention: {outputs_with_attn['video_attention'].shape}")
            if 'cross_modal_attention' in outputs_with_attn:
                print(f"   ✓ Cross-modal attention layers: {len(outputs_with_attn['cross_modal_attention'])}")
            if 'temporal_attention' in outputs_with_attn:
                print(f"   ✓ Temporal attention: {outputs_with_attn['temporal_attention'].shape}")
            if 'fusion_weights' in outputs_with_attn:
                print(f"   ✓ Fusion weights: {outputs_with_attn['fusion_weights'].shape}")
                avg_weights = outputs_with_attn['fusion_weights'].mean(dim=[0, 1])
                print(f"      Audio: {avg_weights[0]:.4f}, Video: {avg_weights[1]:.4f}")
        except Exception as e:
            print(f"   ✗ Attention weights failed: {e}")
    else:
        print(f"   - Not supported in {model_name}")

    # 4. 损失计算
    print(f"\n4. Loss Computation:")
    targets = {
        'blendshapes': torch.rand_like(outputs['blendshapes']),
        'head_pose': F.normalize(torch.randn_like(outputs['head_pose']), dim=-1),
    }

    try:
        losses = model.compute_loss(outputs, targets)
        print(f"   ✓ Loss computation successful")
        for name, value in losses.items():
            print(f"      {name}: {value.item():.6f}")
    except Exception as e:
        print(f"   ✗ Loss computation failed: {e}")

    # 5. 输出统计
    print(f"\n5. Output Statistics:")
    print(f"   BlendShapes:")
    print(f"      min: {outputs['blendshapes'].min().item():.4f}")
    print(f"      max: {outputs['blendshapes'].max().item():.4f}")
    print(f"      mean: {outputs['blendshapes'].mean().item():.4f}")
    print(f"   Head pose (should be normalized):")
    norms = torch.norm(outputs['head_pose'], dim=-1)
    print(f"      norm min: {norms.min().item():.4f}")
    print(f"      norm max: {norms.max().item():.4f}")
    print(f"      norm mean: {norms.mean().item():.4f}")

    return {
        'num_params': num_params,
        'outputs': outputs,
    }


def run_comparison():
    """运行完整对比测试"""

    print("\n" + "="*80)
    print("DualFusionModel Version Comparison")
    print("="*80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 生成测试数据
    batch_size, seq_len = 4, 100
    audio = torch.randn(batch_size, seq_len, 80).to(device)
    video = torch.randn(batch_size, seq_len, 478 * 3).to(device)

    print(f"Test data shape:")
    print(f"  Audio: {audio.shape}")
    print(f"  Video: {video.shape}")

    # 创建三个版本的模型
    models = {
        'V1 (Original)': create_dual_fusion_model().to(device),
        'V2 (Minimal)': create_dual_fusion_model_v2().to(device),
        'V3 (Full)': create_dual_fusion_model_v3().to(device),
    }

    # 测试所有模型
    results = {}
    for name, model in models.items():
        result = test_model_features(name.split()[0], model, audio, video)
        if result:
            results[name] = result

    # 性能对比
    print("\n" + "="*80)
    print("Performance Comparison")
    print("="*80)

    timing_results = {}
    for name, model in models.items():
        print(f"\nMeasuring {name}...")
        timing = measure_inference_time(
            model, audio, video,
            num_runs=50, warmup=10, device=device
        )
        timing_results[name] = timing
        print(f"  Mean: {timing['mean_ms']:.2f} ms")
        print(f"  Std: {timing['std_ms']:.2f} ms")
        print(f"  Min: {timing['min_ms']:.2f} ms")
        print(f"  Max: {timing['max_ms']:.2f} ms")

    # 汇总对比表
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)

    # 表头
    header = f"{'Version':<20} {'Params':<15} {'Mean Time':<15} {'Features':<30}"
    print(header)
    print("-" * len(header))

    for name in models.keys():
        params = results[name]['num_params']
        params_str = f"{params/1e6:.2f}M"

        timing = timing_results[name]
        time_str = f"{timing['mean_ms']:.2f}ms"

        # 特性列表
        features = []
        if 'V1' in name:
            features = ['Basic']
        elif 'V2' in name:
            features = ['FFN', 'Temporal', 'Attn']
        elif 'V3' in name:
            features = ['Multi-layer', 'Adaptive', 'Full']

        features_str = ', '.join(features)

        print(f"{name:<20} {params_str:<15} {time_str:<15} {features_str:<30}")

    # 详细特性对比
    print("\n" + "="*80)
    print("Feature Comparison")
    print("="*80)

    feature_table = [
        ['Feature', 'V1', 'V2', 'V3'],
        ['-' * 30, '-' * 10, '-' * 10, '-' * 10],
        ['Cross-modal Attention', '✓', '✓', '✓'],
        ['FFN after Attention', '✗', '✓', '✓'],
        ['Multi-layer Cross-modal', '✗', '✗', '✓'],
        ['Original Feature Preserved', '✗', '✓', '✗'],
        ['Enhanced Fusion', '✗', '✓', '✓'],
        ['Adaptive Fusion Gate', '✗', '✗', '✓'],
        ['Temporal LSTM', '✗', '✓', '✓ (2-layer)'],
        ['Temporal Attention', '✗', '✗', '✓'],
        ['Attention Visualization', '✗', '✓', '✓'],
        ['Fusion Weight Visualization', '✗', '✗', '✓'],
    ]

    for row in feature_table:
        print(f"{row[0]:<30} {row[1]:<10} {row[2]:<10} {row[3]:<10}")

    # 改进建议
    print("\n" + "="*80)
    print("Recommendations")
    print("="*80)

    print("\n1. 选择V1（原始版）如果：")
    print("   - 需要最快的推理速度")
    print("   - 模型参数量受限")
    print("   - 只需要基础功能")

    print("\n2. 选择V2（最小改动）如果：")
    print("   - 追求性能和速度的平衡")
    print("   - 需要注意力可视化功能")
    print("   - 想要快速改进效果")

    print("\n3. 选择V3（完整改进）如果：")
    print("   - 追求最佳模型性能")
    print("   - 不限制模型大小")
    print("   - 需要深入分析（融合权重等）")
    print("   - 用于研究和实验")

    print("\n" + "="*80)


if __name__ == '__main__':
    try:
        run_comparison()
        print("\n✓ All comparisons completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
