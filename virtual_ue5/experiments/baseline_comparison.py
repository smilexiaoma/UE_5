"""
Baseline对比实验
比较不同baseline模型的性能：
- Base Audio Only
- Base Video Only
"""

import os
import sys
import subprocess
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_experiment(model_name, actor=None, epochs=50, batch_size=8):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"运行实验: {model_name}")
    if actor:
        print(f"Actor: {actor}")
    print(f"{'='*60}\n")

    # 构建命令
    cmd = [
        'python', '../train_ravdess.py',
        '--model', model_name,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--save_dir', f'../results/baseline_comparison/{model_name}'
    ]

    if actor:
        cmd.extend(['--actor', actor])
        cmd[-1] = f'../results/baseline_comparison/{model_name}_{actor}'

    # 运行实验
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ 实验完成: {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 实验失败: {model_name}")
        print(f"错误: {e}")
        return False

    return True


def main():
    print("\n" + "="*60)
    print("Baseline模型对比实验")
    print("="*60)

    # 实验配置
    models = ['base_audio', 'base_video']
    epochs = 50
    batch_size = 8

    # 可选：仅使用Actor_01进行快速验证
    use_single_actor = True
    actor = 'Actor_01' if use_single_actor else None

    print(f"\n配置:")
    print(f"  模型: {models}")
    print(f"  训练轮数: {epochs}")
    print(f"  批大小: {batch_size}")
    print(f"  Actor: {actor if actor else '全部'}")

    # 运行所有实验
    results = {}
    for model in models:
        success = run_experiment(model, actor=actor, epochs=epochs, batch_size=batch_size)
        results[model] = 'Success' if success else 'Failed'

    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for model, status in results.items():
        print(f"  {model}: {status}")
    print("="*60)


if __name__ == '__main__':
    main()
