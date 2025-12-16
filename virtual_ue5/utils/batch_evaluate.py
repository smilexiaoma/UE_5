"""
批量评估所有训练好的模型
生成详细的评估结果和对比分析
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

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


def evaluate_single_model(model, dataloader, device, model_name):
    """评估单个模型"""
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

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估进度", leave=False):
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

    # 计算相关系数 (Correlation)
    pred_flat = all_pred_blendshapes.reshape(-1, all_pred_blendshapes.shape[-1])
    true_flat = all_true_blendshapes.reshape(-1, all_true_blendshapes.shape[-1])

    correlations = []
    for i in range(pred_flat.shape[1]):
        corr = np.corrcoef(pred_flat[:, i], true_flat[:, i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    results['avg_correlation'] = np.mean(correlations) if correlations else 0.0

    # 计算每个 blendshape 的平均误差
    per_blendshape_mae = np.mean(np.abs(all_pred_blendshapes - all_true_blendshapes), axis=(0, 1))
    results['per_blendshape_mae'] = per_blendshape_mae.tolist()

    return results, all_pred_blendshapes, all_true_blendshapes


def save_visualizations(pred_blendshapes, true_blendshapes, results, save_dir):
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

    # 2. 绘制预测 vs 真实值散点图
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


def find_model_checkpoints(results_dir):
    """查找所有模型checkpoint"""
    model_paths = []

    for root, dirs, files in os.walk(results_dir):
        if 'best_model.pt' in files:
            config_path = Path(root) / 'config.json'
            if config_path.exists():
                model_paths.append({
                    'checkpoint': Path(root) / 'best_model.pt',
                    'config': config_path,
                    'dir': Path(root)
                })

    return model_paths


def main():
    parser = argparse.ArgumentParser(description="批量评估所有训练好的模型")
    parser.add_argument('--results_dir', type=str, default='results',
                       help='结果目录路径')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/digietal_data/1188976',
                       help='RAVDESS数据集路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')

    args = parser.parse_args()

    # 查找所有模型
    print("=" * 80)
    print("批量评估模型")
    print("=" * 80)

    model_paths = find_model_checkpoints(args.results_dir)

    # 过滤只保留baseline_training和innovation_training
    model_paths = [m for m in model_paths if 'baseline_training' in str(m['dir']) or 'innovation_training' in str(m['dir'])]

    print(f"\n找到 {len(model_paths)} 个训练好的模型")

    device = get_device()
    print(f"使用设备: {device}\n")

    # 存储所有评估结果
    all_results = []

    # 评估每个模型
    for i, model_info in enumerate(model_paths, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(model_paths)}] 评估模型: {model_info['dir'].name}")
        print(f"{'='*80}")

        # 加载配置
        with open(model_info['config'], 'r') as f:
            config = json.load(f)

        print(f"模型类型: {config['model']}")
        print(f"Actor: {config.get('actor', '全部演员')}")

        # 创建数据集
        actors = [config['actor']] if config.get('actor') else None

        try:
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
            model = create_model(config['model'])
            model = model.to(device)

            # 加载模型权重
            checkpoint = torch.load(model_info['checkpoint'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"训练Epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"训练验证损失: {checkpoint.get('val_loss', 'Unknown'):.6f}")

            # 评估模型
            results, pred_bs, true_bs = evaluate_single_model(
                model, dataloader, device, config['model']
            )

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

            # 添加模型信息
            results_serializable['model_name'] = config['model']
            results_serializable['actor'] = config.get('actor', 'all_actors')
            results_serializable['dir_name'] = model_info['dir'].name
            results_serializable['train_val_loss'] = float(checkpoint.get('val_loss', 0))

            # 保存评估结果
            results_path = model_info['dir'] / 'evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)

            print(f"\n评估指标:")
            print(f"  Blendshape MAE:  {results['blendshape_mae']:.6f}")
            print(f"  Blendshape RMSE: {results['blendshape_rmse']:.6f}")
            print(f"  Head Pose MAE:   {results['head_pose_mae']:.6f}")
            print(f"  Head Pose RMSE:  {results['head_pose_rmse']:.6f}")
            print(f"  Correlation:     {results['avg_correlation']:.4f}")

            # 保存可视化
            save_visualizations(pred_bs, true_bs, results, model_info['dir'])
            print(f"✓ 评估结果已保存至: {results_path}")

            # 添加到总结果
            all_results.append(results_serializable)

            # 清理内存
            del model, dataset, dataloader
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ 评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 生成对比报告
    print(f"\n{'='*80}")
    print("生成对比报告")
    print(f"{'='*80}\n")

    # 保存综合结果
    summary_path = Path(args.results_dir) / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ 综合评估结果已保存至: {summary_path}")

    # 生成对比表格
    df_data = []
    for r in all_results:
        df_data.append({
            '模型': r['model_name'],
            'Actor': r['actor'],
            'Blendshape MAE': f"{r['blendshape_mae']:.6f}",
            'Blendshape RMSE': f"{r['blendshape_rmse']:.6f}",
            'Head Pose MAE': f"{r['head_pose_mae']:.6f}",
            'Head Pose RMSE': f"{r['head_pose_rmse']:.6f}",
            'Correlation': f"{r['avg_correlation']:.4f}",
        })

    df = pd.DataFrame(df_data)

    print("\n模型对比表:")
    print(df.to_string(index=False))

    # 保存CSV
    csv_path = Path(args.results_dir) / 'evaluation_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 对比表格已保存至: {csv_path}")

    print(f"\n{'='*80}")
    print("批量评估完成!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
