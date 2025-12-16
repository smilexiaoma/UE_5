"""
全数据集训练实验
使用RAVDESS全部Actor的数据进行训练
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_full_dataset_training(model_name, epochs=100, batch_size=16):
    """使用全数据集训练模型"""
    print(f"\n{'='*60}")
    print(f"全数据集训练实验")
    print(f"模型: {model_name}")
    print(f"使用全部Actors")
    print(f"{'='*60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'../results/full_dataset/{model_name}_all_actors_{timestamp}'

    # 构建命令（不指定--actor则使用全部数据）
    cmd = [
        'python', '../train_ravdess.py',
        '--model', model_name,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--save_dir', save_dir,
        '--num_workers', '4'  # 使用多进程加速数据加载
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
    print("全数据集训练实验")
    print("="*60)

    # 实验配置
    config = {
        'model': 'base_audio',  # 可以改为: base_video, dual_fusion, diffusion, e2e_loop
        'epochs': 100,
        'batch_size': 16,  # 根据GPU内存调整
        'dataset': 'RAVDESS全部Actors'
    }

    print(f"\n实验配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\n注意事项:")
    print(f"  - 全数据集训练需要较长时间")
    print(f"  - 建议使用GPU训练")
    print(f"  - 根据GPU内存调整batch_size")

    # 确认是否继续
    response = input("\n是否开始训练? (y/n): ")
    if response.lower() != 'y':
        print("训练已取消")
        return

    # 运行实验
    success, save_dir = run_full_dataset_training(
        model_name=config['model'],
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
