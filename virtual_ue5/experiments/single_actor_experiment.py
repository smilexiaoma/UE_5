"""
单个Actor训练实验
验证实验的完整性，仅使用Actor_01进行训练
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_single_actor_training(model_name, actor='Actor_01', epochs=50, batch_size=8):
    """使用单个Actor训练模型"""
    print(f"\n{'='*60}")
    print(f"单个Actor训练实验")
    print(f"模型: {model_name}")
    print(f"Actor: {actor}")
    print(f"{'='*60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'../results/single_actor/{model_name}_{actor}_{timestamp}'

    # 构建命令
    cmd = [
        'python', '../train_ravdess.py',
        '--model', model_name,
        '--actor', actor,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--save_dir', save_dir
    ]

    print(f"命令: {' '.join(cmd)}\n")

    # 运行训练
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ 训练完成")
        print(f"结果保存至: {save_dir}")
        return True, save_dir
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 训练失败")
        print(f"错误: {e}")
        return False, None


def main():
    print("\n" + "="*60)
    print("单个Actor训练实验（验证实验完整性）")
    print("="*60)

    # 实验配置
    config = {
        'model': 'base_audio',  # 可以改为其他模型
        'actor': 'Actor_01',
        'epochs': 50,
        'batch_size': 8
    }

    print(f"\n实验配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 运行实验
    success, save_dir = run_single_actor_training(
        model_name=config['model'],
        actor=config['actor'],
        epochs=config['epochs'],
        batch_size=config['batch_size']
    )

    # 保存实验配置
    if success and save_dir:
        config_file = os.path.join(save_dir, 'experiment_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n实验配置已保存: {config_file}")

    print("\n" + "="*60)
    print(f"实验状态: {'成功' if success else '失败'}")
    print("="*60)


if __name__ == '__main__':
    main()
