"""
消融实验 (Ablation Study)
系统地移除或替换模型的某些组件，以评估各组件的贡献

实验内容：
1. 移除区域约束（use_region_constraints=False）
2. 移除时间一致性损失（temporal_loss权重=0）
3. 移除表情强度正则化（intensity_loss权重=0）
4. 移除head pose预测（predict_head_pose=False）
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.baseline.base_model import create_base_model
from utils.common import set_seed, get_device, count_parameters, AverageMeter
from utils.ravdess_dataset import RAVDESSDataset


def train_model_variant(
    variant_name,
    model_config,
    loss_weights,
    epochs=30,
    batch_size=8,
    actor='Actor_01'
):
    """训练模型的某个变体"""
    print(f"\n{'='*60}")
    print(f"消融实验变体: {variant_name}")
    print(f"{'='*60}")

    # 设置设备和随机种子
    device = get_device()
    set_seed(42)

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'../results/ablation_study/{variant_name}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    config = {
        'variant_name': variant_name,
        'model_config': model_config,
        'loss_weights': loss_weights,
        'epochs': epochs,
        'batch_size': batch_size,
        'actor': actor
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 创建数据集
    print(f"\n加载数据集 (Actor: {actor})...")
    train_dataset = RAVDESSDataset(
        data_dir='../data/1188976',
        actors=[actor],
        max_samples=None,
        seq_len=100,
        use_cache=True,
    )

    # 划分训练和验证集
    val_size = max(1, len(train_dataset) // 5)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # 创建模型
    print(f"\n创建模型...")
    model = create_base_model(mode='audio', config=model_config)
    model = model.to(device)
    print(f"参数量: {count_parameters(model):,}")

    # 优化器
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss_meter = AverageMeter()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            audio = batch['audio'].to(device)
            targets = {
                'blendshapes': batch['blendshapes'].to(device),
                'head_pose': batch['head_pose'].to(device),
            }

            optimizer.zero_grad()
            outputs = model(audio=audio)
            losses = model.compute_loss(outputs, targets, weights=loss_weights)

            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_meter.update(losses['total_loss'].item())

        # 验证
        model.eval()
        val_loss_meter = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(device)
                targets = {
                    'blendshapes': batch['blendshapes'].to(device),
                    'head_pose': batch['head_pose'].to(device),
                }

                outputs = model(audio=audio)
                losses = model.compute_loss(outputs, targets, weights=loss_weights)
                val_loss_meter.update(losses['total_loss'].item())

        # 记录
        train_loss = train_loss_meter.avg
        val_loss = val_loss_meter.avg
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, os.path.join(save_dir, 'best_model.pt'))

        scheduler.step()

    # 保存训练历史
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ 训练完成")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"结果保存至: {save_dir}")

    return best_val_loss


def main():
    print("\n" + "="*60)
    print("消融实验 (Ablation Study)")
    print("="*60)

    # 完整模型（基线）
    full_model = {
        'use_region_constraints': True,
        'predict_head_pose': True,
    }
    full_weights = {
        'blendshape': 1.0,
        'head_pose': 0.5,
        'temporal': 0.1,
        'intensity': 0.01,
    }

    # 定义所有消融变体
    variants = {
        'full_model': {
            'config': full_model,
            'weights': full_weights,
            'description': '完整模型（包含所有组件）'
        },
        'no_region_constraints': {
            'config': {**full_model, 'use_region_constraints': False},
            'weights': full_weights,
            'description': '移除区域约束'
        },
        'no_temporal_loss': {
            'config': full_model,
            'weights': {**full_weights, 'temporal': 0.0},
            'description': '移除时间一致性损失'
        },
        'no_intensity_regularization': {
            'config': full_model,
            'weights': {**full_weights, 'intensity': 0.0},
            'description': '移除表情强度正则化'
        },
        'no_head_pose': {
            'config': {**full_model, 'predict_head_pose': False},
            'weights': {**full_weights, 'head_pose': 0.0},
            'description': '移除头部姿态预测'
        },
    }

    print(f"\n消融实验变体:")
    for name, info in variants.items():
        print(f"  - {name}: {info['description']}")

    # 实验配置
    epochs = 30
    batch_size = 8
    actor = 'Actor_01'

    print(f"\n训练配置:")
    print(f"  训练轮数: {epochs}")
    print(f"  批大小: {batch_size}")
    print(f"  Actor: {actor}")

    # 运行所有变体
    results = {}
    for variant_name, variant_info in variants.items():
        try:
            best_loss = train_model_variant(
                variant_name=variant_name,
                model_config=variant_info['config'],
                loss_weights=variant_info['weights'],
                epochs=epochs,
                batch_size=batch_size,
                actor=actor
            )
            results[variant_name] = best_loss
        except Exception as e:
            print(f"\n✗ 变体 {variant_name} 训练失败: {e}")
            results[variant_name] = None

    # 打印消融实验结果对比
    print("\n" + "="*60)
    print("消融实验结果对比")
    print("="*60)
    for variant_name, loss in results.items():
        if loss is not None:
            print(f"{variant_name:30s}: {loss:.4f}")
        else:
            print(f"{variant_name:30s}: Failed")

    # 保存结果汇总
    results_file = f'../results/ablation_study/summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果汇总已保存: {results_file}")
    print("="*60)


if __name__ == '__main__':
    main()
