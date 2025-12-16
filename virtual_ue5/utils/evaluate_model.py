"""
评估训练好的模型
输出详细的评估指标和可视化
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline.base_model import create_base_model
from src.innovation1_dual_fusion.innovation1_dual_fusion import create_dual_fusion_model
from src.innovation2_diffusion.innovation2_diffusion import create_diffusion_model
from src.innovation3_e2e_loop.innovation3_e2e_loop import create_e2e_loop_model
from utils.common import get_device
from utils.ravdess_dataset import RAVDESSDataset


def create_model(model_name: str, config=None):
    """创建模型"""
    if model_name == 'base_audio':
        return create_base_model(mode='audio', config=config)
    elif model_name == 'base_video':
        return create_base_model(mode='video', config=config)
    elif model_name == 'dual_fusion':
        return create_dual_fusion_model(config=config)
    elif model_name == 'diffusion':
        return create_diffusion_model(config=config)
    elif model_name == 'e2e_loop':
        return create_e2e_loop_model(config=config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model, dataloader, device, model_name):
    """评估模型"""
    model.eval()

    # 用于存储所有预测和真实值
    all_pred_blendshapes = []
    all_true_blendshapes = []
    all_pred_head_pose = []
    all_true_head_pose = []

    # 损失统计
    loss_stats = {
        'total_loss': [],
        'blendshape_loss': [],
        'head_pose_loss': [],
        'temporal_loss': [],
    }

    print("开始评估...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估进度"):
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            blendshapes = batch['blendshapes'].to(device)
            head_pose = batch['head_pose'].to(device)

            targets = {
                'blendshapes': blendshapes,
                'head_pose': head_pose,
            }

            # 前向传播
            if model_name == 'base_audio':
                outputs = model(audio=audio)
            elif model_name == 'base_video':
                outputs = model(video=video)
            elif model_name == 'dual_fusion':
                outputs = model(audio=audio, video=video)
            elif model_name == 'diffusion':
                outputs = model(audio=audio, video=video, blendshapes=blendshapes)
            elif model_name == 'e2e_loop':
                outputs = model(audio=audio, video=video)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # 计算损失
            losses = model.compute_loss(outputs, targets)

            # 记录损失
            for key in loss_stats.keys():
                if key in losses and isinstance(losses[key], torch.Tensor):
                    loss_stats[key].append(losses[key].item())

            # 收集预测和真实值
            all_pred_blendshapes.append(outputs['blendshapes'].cpu().numpy())
            all_true_blendshapes.append(blendshapes.cpu().numpy())
            all_pred_head_pose.append(outputs['head_pose'].cpu().numpy())
            all_true_head_pose.append(head_pose.cpu().numpy())

    # 合并所有结果
    all_pred_blendshapes = np.concatenate(all_pred_blendshapes, axis=0)
    all_true_blendshapes = np.concatenate(all_true_blendshapes, axis=0)
    all_pred_head_pose = np.concatenate(all_pred_head_pose, axis=0)
    all_true_head_pose = np.concatenate(all_true_head_pose, axis=0)

    # 计算评估指标
    results = {}

    # 平均损失
    for key, values in loss_stats.items():
        if values:
            results[f'avg_{key}'] = np.mean(values)

    # Blendshape 评估指标
    blendshape_mae = np.mean(np.abs(all_pred_blendshapes - all_true_blendshapes))
    blendshape_mse = np.mean((all_pred_blendshapes - all_true_blendshapes) ** 2)
    blendshape_rmse = np.sqrt(blendshape_mse)

    results['blendshape_mae'] = blendshape_mae
    results['blendshape_mse'] = blendshape_mse
    results['blendshape_rmse'] = blendshape_rmse

    # Head Pose 评估指标
    head_pose_mae = np.mean(np.abs(all_pred_head_pose - all_true_head_pose))
    head_pose_mse = np.mean((all_pred_head_pose - all_true_head_pose) ** 2)
    head_pose_rmse = np.sqrt(head_pose_mse)

    results['head_pose_mae'] = head_pose_mae
    results['head_pose_mse'] = head_pose_mse
    results['head_pose_rmse'] = head_pose_rmse

    # 计算每个 blendshape 的平均误差
    per_blendshape_mae = np.mean(np.abs(all_pred_blendshapes - all_true_blendshapes), axis=(0, 1))
    results['per_blendshape_mae'] = per_blendshape_mae.tolist()

    return results, all_pred_blendshapes, all_true_blendshapes, all_pred_head_pose, all_true_head_pose


def print_results(results, model_name):
    """打印评估结果"""
    print("\n" + "=" * 80)
    print(f"模型评估结果: {model_name}")
    print("=" * 80)

    print("\n📊 损失指标:")
    print(f"  总损失 (Total Loss):              {results.get('avg_total_loss', 0):.6f}")
    if 'avg_blendshape_loss' in results:
        print(f"  Blendshape损失:                   {results['avg_blendshape_loss']:.6f}")
    if 'avg_head_pose_loss' in results:
        print(f"  头部姿态损失:                     {results['avg_head_pose_loss']:.6f}")
    if 'avg_temporal_loss' in results:
        print(f"  时序一致性损失:                   {results['avg_temporal_loss']:.6f}")

    print("\n📈 Blendshape 评估指标:")
    print(f"  平均绝对误差 (MAE):               {results['blendshape_mae']:.6f}")
    print(f"  均方误差 (MSE):                   {results['blendshape_mse']:.6f}")
    print(f"  均方根误差 (RMSE):                {results['blendshape_rmse']:.6f}")

    print("\n🎯 头部姿态评估指标:")
    print(f"  平均绝对误差 (MAE):               {results['head_pose_mae']:.6f}")
    print(f"  均方误差 (MSE):                   {results['head_pose_mse']:.6f}")
    print(f"  均方根误差 (RMSE):                {results['head_pose_rmse']:.6f}")

    print("\n" + "=" * 80)


def save_visualizations(results, pred_blendshapes, true_blendshapes, save_dir):
    """保存可视化结果"""
    viz_dir = Path(save_dir) / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    # 1. 绘制每个 blendshape 的误差分布
    per_blendshape_mae = results['per_blendshape_mae']

    plt.figure(figsize=(15, 6))
    plt.bar(range(len(per_blendshape_mae)), per_blendshape_mae, color='steelblue', alpha=0.7)
    plt.xlabel('Blendshape Index', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.title('Per-Blendshape MAE Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'per_blendshape_error.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 绘制预测 vs 真实值散点图（采样部分数据）
    sample_size = min(5000, pred_blendshapes.size)
    sample_indices = np.random.choice(pred_blendshapes.size, sample_size, replace=False)

    pred_sample = pred_blendshapes.flatten()[sample_indices]
    true_sample = true_blendshapes.flatten()[sample_indices]

    plt.figure(figsize=(10, 10))
    plt.scatter(true_sample, pred_sample, alpha=0.3, s=1, color='steelblue')

    # 添加理想对角线
    min_val = min(true_sample.min(), pred_sample.min())
    max_val = max(true_sample.max(), pred_sample.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')

    plt.xlabel('True Blendshape Values', fontsize=12)
    plt.ylabel('Predicted Blendshape Values', fontsize=12)
    plt.title('Predicted vs True Blendshapes', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'pred_vs_true_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 绘制误差直方图
    errors = (pred_blendshapes - true_blendshapes).flatten()

    plt.figure(figsize=(12, 6))
    plt.hist(errors, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Error Distribution', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 可视化结果已保存至: {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description="评估训练好的模型")
    parser.add_argument('--model_path', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/digietal_data/1188976',
                       help='RAVDESS数据集路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--save_results', action='store_true', help='保存评估结果')

    args = parser.parse_args()

    # 加载模型配置
    model_dir = Path(args.model_path).parent
    config_path = model_dir / 'config.json'

    if not config_path.exists():
        print(f"错误: 找不到配置文件 {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"加载模型配置:")
    print(f"  模型类型: {config['model']}")
    print(f"  Actor: {config.get('actor', '全部')}")

    # 创建数据集
    print(f"\n加载数据集...")
    actors = [config['actor']] if config.get('actor') else None

    dataset = RAVDESSDataset(
        data_dir=args.data_dir,
        actors=actors,
        seq_len=config.get('seq_len', 100),
        use_cache=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"数据集样本数: {len(dataset)}")

    # 创建模型
    device = get_device()
    print(f"\n使用设备: {device}")

    model = create_model(config['model'])
    model = model.to(device)

    # 加载模型权重
    print(f"加载模型权重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"模型epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"训练验证损失: {checkpoint.get('val_loss', 'Unknown')}")

    # 评估模型
    results, pred_bs, true_bs, pred_hp, true_hp = evaluate_model(
        model, dataloader, device, config['model']
    )

    # 打印结果
    print_results(results, config['model'])

    # 保存结果
    if args.save_results:
        # 转换numpy类型为Python原生类型
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, (np.floating, np.integer)):
                results_serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, list):
                results_serializable[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
            else:
                results_serializable[key] = value

        results_path = model_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\n✓ 评估结果已保存至: {results_path}")

        # 保存可视化
        save_visualizations(results, pred_bs, true_bs, model_dir)


if __name__ == '__main__':
    main()
