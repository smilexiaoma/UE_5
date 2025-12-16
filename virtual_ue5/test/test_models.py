"""
模型测试与对比脚本
比较所有模型的性能指标

使用方法：
python test_models.py                    # 运行所有模型测试
python test_models.py --model dual_fusion  # 只测试特定模型
"""

import os
import sys
import argparse
import time
from typing import Dict, List
import json

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline.base_model import create_base_model
from src.innovation1_dual_fusion.innovation1_dual_fusion import create_dual_fusion_model
from src.innovation2_diffusion.innovation2_diffusion import create_diffusion_model
from src.innovation3_e2e_loop.innovation3_e2e_loop import create_e2e_loop_model
from utils.common import set_seed, get_device, count_parameters


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, device: torch.device):
        self.device = device

    def compute_metrics(
        self,
        pred_blendshapes: torch.Tensor,
        target_blendshapes: torch.Tensor,
    ) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            pred_blendshapes: (B, T, 52) 预测的BlendShape
            target_blendshapes: (B, T, 52) 目标BlendShape
        """
        metrics = {}

        # MSE
        mse = F.mse_loss(pred_blendshapes, target_blendshapes).item()
        metrics['mse'] = mse

        # RMSE
        metrics['rmse'] = np.sqrt(mse)

        # MAE
        mae = F.l1_loss(pred_blendshapes, target_blendshapes).item()
        metrics['mae'] = mae

        # 相关系数 (Pearson) 
        pred_flat = pred_blendshapes.view(-1).cpu().numpy()
        target_flat = target_blendshapes.view(-1).cpu().numpy()
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        metrics['correlation'] = correlation

        # 口型准确率 (LVE - Lip Vertex Error)
        mouth_indices = list(range(17, 42))  # 口部BlendShape索引
        mouth_pred = pred_blendshapes[:, :, mouth_indices]
        mouth_target = target_blendshapes[:, :, mouth_indices]
        lve = F.mse_loss(mouth_pred, mouth_target).item()
        metrics['lip_error'] = lve

        # 时间平滑度 (Jitter)
        pred_diff = pred_blendshapes[:, 1:] - pred_blendshapes[:, :-1]
        jitter = (pred_diff ** 2).mean().item()
        metrics['jitter'] = jitter

        # 表情范围 (动态范围)
        pred_range = (pred_blendshapes.max(dim=1)[0] - pred_blendshapes.min(dim=1)[0]).mean().item()
        metrics['dynamic_range'] = pred_range

        return metrics

    def measure_inference_time(
        self,
        model: torch.nn.Module,
        audio: torch.Tensor,
        video: torch.Tensor,
        model_name: str,
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        测量推理时间

        Returns:
            dict: {'mean_ms': float, 'std_ms': float, 'fps': float}
        """
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                if model_name == 'base_audio':
                    _ = model(audio=audio)
                elif model_name == 'base_video':
                    _ = model(video=video)
                elif model_name == 'dual_fusion':
                    _ = model(audio=audio, video=video)
                elif model_name == 'diffusion':
                    _ = model.sample(audio=audio, video=video, num_inference_steps=10)
                elif model_name == 'e2e_loop':
                    _ = model(audio=audio, video=video)

        # 同步CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # 计时
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()

                if model_name == 'base_audio':
                    _ = model(audio=audio)
                elif model_name == 'base_video':
                    _ = model(video=video)
                elif model_name == 'dual_fusion':
                    _ = model(audio=audio, video=video)
                elif model_name == 'diffusion':
                    _ = model.sample(audio=audio, video=video, num_inference_steps=10)
                elif model_name == 'e2e_loop':
                    _ = model(audio=audio, video=video)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)  # 转换为毫秒

        mean_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / mean_time if mean_time > 0 else 0

        return {
            'mean_ms': mean_time,
            'std_ms': std_time,
            'fps': fps,
        }


def generate_test_data(
    batch_size: int = 4,
    seq_len: int = 100,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """生成测试数据"""
    audio = torch.randn(batch_size, seq_len, 80, device=device)
    video = torch.randn(batch_size, seq_len, 478 * 3, device=device) * 0.1 + 0.5
    blendshapes = torch.sigmoid(torch.randn(batch_size, seq_len, 52, device=device))
    head_pose = F.normalize(torch.randn(batch_size, seq_len, 4, device=device), dim=-1)

    return {
        'audio': audio,
        'video': video,
        'blendshapes': blendshapes,
        'head_pose': head_pose,
    }


def test_model(
    model_name: str,
    device: torch.device,
    evaluator: ModelEvaluator,
    test_data: Dict[str, torch.Tensor],
    verbose: bool = True,
) -> Dict[str, any]:
    """测试单个模型"""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print('='*60)

    # 创建模型
    if model_name == 'base_audio':
        model = create_base_model(mode='audio')
    elif model_name == 'base_video':
        model = create_base_model(mode='video')
    elif model_name == 'dual_fusion':
        model = create_dual_fusion_model()
    elif model_name == 'diffusion':
        model = create_diffusion_model()
    elif model_name == 'e2e_loop':
        model = create_e2e_loop_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    model.eval()

    # 模型信息
    num_params = count_parameters(model)
    if verbose:
        print(f"Parameters: {num_params:,}")

    # 前向传播
    audio = test_data['audio']
    video = test_data['video']
    target_blendshapes = test_data['blendshapes']

    with torch.no_grad():
        if model_name == 'base_audio':
            outputs = model(audio=audio)
        elif model_name == 'base_video':
            outputs = model(video=video)
        elif model_name == 'dual_fusion':
            outputs = model(audio=audio, video=video)
        elif model_name == 'diffusion':
            outputs = model.sample(audio=audio, video=video, num_inference_steps=10)
        elif model_name == 'e2e_loop':
            outputs = model(audio=audio, video=video)

    pred_blendshapes = outputs['blendshapes']

    # 计算指标
    metrics = evaluator.compute_metrics(pred_blendshapes, target_blendshapes)

    if verbose:
        print(f"\nQuality Metrics:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  Lip Error: {metrics['lip_error']:.6f}")
        print(f"  Jitter: {metrics['jitter']:.6f}")
        print(f"  Dynamic Range: {metrics['dynamic_range']:.4f}")

    # 测量推理时间
    timing = evaluator.measure_inference_time(
        model, audio, video, model_name,
        num_runs=50,
    )

    if verbose:
        print(f"\nTiming (batch={audio.shape[0]}, seq_len={audio.shape[1]}):")
        print(f"  Mean: {timing['mean_ms']:.2f} ms")
        print(f"  Std: {timing['std_ms']:.2f} ms")
        print(f"  FPS: {timing['fps']:.1f}")

    return {
        'model_name': model_name,
        'num_params': num_params,
        'metrics': metrics,
        'timing': timing,
    }


def run_comparison(
    models: List[str],
    device: torch.device,
    batch_size: int = 4,
    seq_len: int = 100,
) -> Dict[str, any]:
    """运行模型对比"""

    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)

    # 生成测试数据
    test_data = generate_test_data(batch_size, seq_len, device)

    # 创建评估器
    evaluator = ModelEvaluator(device)

    # 测试所有模型
    results = {}
    for model_name in models:
        try:
            result = test_model(model_name, device, evaluator, test_data)
            results[model_name] = result
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue

    # 打印对比表格
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)

    # 表头
    headers = ['Model', 'Params', 'MSE', 'MAE', 'Correlation', 'Lip Error', 'Jitter', 'Time(ms)', 'FPS']
    header_str = f"{'Model':<15} {'Params':<12} {'MSE':<10} {'MAE':<10} {'Corr':<10} {'Lip':<10} {'Jitter':<10} {'Time':<10} {'FPS':<8}"
    print(header_str)
    print("-" * len(header_str))

    for model_name, result in results.items():
        metrics = result['metrics']
        timing = result['timing']
        params_str = f"{result['num_params']/1e6:.2f}M"

        row = f"{model_name:<15} {params_str:<12} {metrics['mse']:<10.6f} {metrics['mae']:<10.6f} " \
              f"{metrics['correlation']:<10.4f} {metrics['lip_error']:<10.6f} {metrics['jitter']:<10.6f} " \
              f"{timing['mean_ms']:<10.2f} {timing['fps']:<8.1f}"
        print(row)

    return results


def test_single_model_detail(model_name: str, device: torch.device):
    """详细测试单个模型"""

    print(f"\n{'='*60}")
    print(f"Detailed Test: {model_name}")
    print('='*60)

    # 创建模型
    if model_name == 'base_audio':
        model = create_base_model(mode='audio')
    elif model_name == 'base_video':
        model = create_base_model(mode='video')
    elif model_name == 'dual_fusion':
        model = create_dual_fusion_model()
    elif model_name == 'diffusion':
        model = create_diffusion_model()
    elif model_name == 'e2e_loop':
        model = create_e2e_loop_model()

    model = model.to(device)

    # 打印模型结构
    print("\nModel Architecture:")
    print(model)

    # 生成测试数据
    test_data = generate_test_data(4, 100, device)

    # 测试不同序列长度的性能
    print("\nSequence Length vs Performance:")
    seq_lengths = [25, 50, 100, 200]
    evaluator = ModelEvaluator(device)

    for seq_len in seq_lengths:
        test_data_var = generate_test_data(4, seq_len, device)
        timing = evaluator.measure_inference_time(
            model,
            test_data_var['audio'],
            test_data_var['video'],
            model_name,
            num_runs=20,
        )
        print(f"  seq_len={seq_len}: {timing['mean_ms']:.2f}ms, {timing['fps']:.1f} FPS")

    # 测试不同batch size的性能
    print("\nBatch Size vs Performance:")
    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        test_data_var = generate_test_data(batch_size, 100, device)
        timing = evaluator.measure_inference_time(
            model,
            test_data_var['audio'],
            test_data_var['video'],
            model_name,
            num_runs=20,
        )
        print(f"  batch_size={batch_size}: {timing['mean_ms']:.2f}ms, {timing['fps']:.1f} FPS")


def main():
    parser = argparse.ArgumentParser(description="Test and compare models")
    parser.add_argument('--model', type=str, default=None,
                        choices=['base_audio', 'base_video', 'dual_fusion', 'diffusion', 'e2e_loop'],
                        help='Specific model to test (default: all)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--detail', action='store_true', help='Run detailed test')
    parser.add_argument('--save', type=str, default=None, help='Save results to JSON file')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 获取设备
    device = get_device()
    print(f"Using device: {device}")

    if args.model:
        if args.detail:
            # 详细测试单个模型
            test_single_model_detail(args.model, device)
        else:
            # 快速测试单个模型
            test_data = generate_test_data(args.batch_size, args.seq_len, device)
            evaluator = ModelEvaluator(device)
            test_model(args.model, device, evaluator, test_data)
    else:
        # 对比所有模型
        all_models = ['base_audio', 'base_video', 'dual_fusion', 'diffusion', 'e2e_loop']
        results = run_comparison(all_models, device, args.batch_size, args.seq_len)

        # 保存结果
        if args.save:
            # 转换为可序列化格式
            save_results = {}
            for model_name, result in results.items():
                save_results[model_name] = {
                    'num_params': result['num_params'],
                    'metrics': result['metrics'],
                    'timing': result['timing'],
                }

            with open(args.save, 'w') as f:
                json.dump(save_results, f, indent=2)
            print(f"\nResults saved to: {args.save}")


if __name__ == '__main__':
    main()
