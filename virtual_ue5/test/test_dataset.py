"""
测试 RAVDESS 数据集加载
验证视频处理、音频提取、BlendShape 估计等功能
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ravdess_dataset import RAVDESSDataset


def test_single_video(data_dir: str, actor: str = 'Actor_01'):
    """测试单个视频的处理"""
    print("=" * 60)
    print("测试单个视频处理")
    print("=" * 60)

    # 创建数据集 (只加载一个样本)
    dataset = RAVDESSDataset(
        data_dir=data_dir,
        actors=[actor],
        max_samples=1,
        seq_len=100,
        use_cache=False,  # 不使用缓存以测试完整流程
    )

    if len(dataset) == 0:
        print(f"错误: 未找到 {actor} 的视频文件")
        return

    # 获取第一个样本
    sample = dataset[0]

    print(f"\n样本信息:")
    print(f"  视频路径: {sample['video_path']}")
    print(f"  情绪: {sample['emotion_name']} (ID: {sample['emotion'].item()})")
    print(f"\n数据形状:")
    print(f"  音频特征: {sample['audio'].shape}")
    print(f"  视频特征: {sample['video'].shape}")
    print(f"  BlendShape: {sample['blendshapes'].shape}")
    print(f"  头部姿态: {sample['head_pose'].shape}")

    print(f"\n数据统计:")
    print(f"  音频 - min: {sample['audio'].min():.4f}, max: {sample['audio'].max():.4f}, mean: {sample['audio'].mean():.4f}")
    print(f"  视频 - min: {sample['video'].min():.4f}, max: {sample['video'].max():.4f}, mean: {sample['video'].mean():.4f}")
    print(f"  BlendShape - min: {sample['blendshapes'].min():.4f}, max: {sample['blendshapes'].max():.4f}, mean: {sample['blendshapes'].mean():.4f}")
    print(f"  头部姿态 - min: {sample['head_pose'].min():.4f}, max: {sample['head_pose'].max():.4f}, mean: {sample['head_pose'].mean():.4f}")

    print("\n✓ 单个视频处理成功!")


def test_actor_dataset(data_dir: str, actor: str = 'Actor_01', max_samples: int = 10):
    """测试加载单个 Actor 的数据集"""
    print("\n" + "=" * 60)
    print(f"测试 {actor} 数据集加载")
    print("=" * 60)

    # 创建数据集
    dataset = RAVDESSDataset(
        data_dir=data_dir,
        actors=[actor],
        max_samples=max_samples,
        seq_len=100,
        use_cache=True,  # 使用缓存加速
    )

    print(f"\n数据集信息:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  情绪分布: {dataset.get_emotion_distribution()}")

    # 测试 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    print(f"\n测试 DataLoader:")
    for i, batch in enumerate(dataloader):
        print(f"  Batch {i + 1}:")
        print(f"    音频: {batch['audio'].shape}")
        print(f"    视频: {batch['video'].shape}")
        print(f"    BlendShape: {batch['blendshapes'].shape}")
        print(f"    头部姿态: {batch['head_pose'].shape}")
        print(f"    情绪: {batch['emotion']}")

        if i >= 1:  # 只测试前2个batch
            break

    print("\n✓ 数据集加载成功!")


def test_full_dataset(data_dir: str, max_samples: int = 50):
    """测试加载完整数据集 (所有 Actor)"""
    print("\n" + "=" * 60)
    print("测试完整数据集加载")
    print("=" * 60)

    # 创建数据集
    dataset = RAVDESSDataset(
        data_dir=data_dir,
        actors=None,  # 加载所有 Actor
        max_samples=max_samples,
        seq_len=100,
        use_cache=True,
    )

    print(f"\n数据集信息:")
    print(f"  总样本数: {len(dataset)}")

    emotion_dist = dataset.get_emotion_distribution()
    print(f"\n  情绪分布:")
    for emotion, count in sorted(emotion_dist.items()):
        print(f"    {emotion}: {count}")

    # 测试批处理
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    print(f"\n测试批处理:")
    batch = next(iter(dataloader))
    print(f"  Batch 形状:")
    print(f"    音频: {batch['audio'].shape}")
    print(f"    视频: {batch['video'].shape}")
    print(f"    BlendShape: {batch['blendshapes'].shape}")
    print(f"    头部姿态: {batch['head_pose'].shape}")

    print("\n✓ 完整数据集加载成功!")


def test_emotion_filtering(data_dir: str, emotions: list):
    """测试情绪过滤功能"""
    print("\n" + "=" * 60)
    print(f"测试情绪过滤: {emotions}")
    print("=" * 60)

    dataset = RAVDESSDataset(
        data_dir=data_dir,
        actors=['Actor_01'],
        emotions=emotions,
        max_samples=20,
        seq_len=100,
        use_cache=True,
    )

    print(f"\n数据集信息:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  情绪分布: {dataset.get_emotion_distribution()}")

    print("\n✓ 情绪过滤成功!")


def benchmark_loading_speed(data_dir: str, num_samples: int = 10):
    """测试数据加载速度"""
    import time

    print("\n" + "=" * 60)
    print("测试数据加载速度")
    print("=" * 60)

    # 测试不使用缓存
    print(f"\n不使用缓存 (处理 {num_samples} 个样本):")
    start = time.time()
    dataset_no_cache = RAVDESSDataset(
        data_dir=data_dir,
        actors=['Actor_01'],
        max_samples=num_samples,
        use_cache=False,
    )
    time_no_cache = time.time() - start
    print(f"  耗时: {time_no_cache:.2f} 秒")
    print(f"  平均: {time_no_cache / num_samples:.2f} 秒/样本")

    # 测试使用缓存
    print(f"\n使用缓存 (加载 {num_samples} 个样本):")
    start = time.time()
    dataset_cache = RAVDESSDataset(
        data_dir=data_dir,
        actors=['Actor_01'],
        max_samples=num_samples,
        use_cache=True,
    )
    time_cache = time.time() - start
    print(f"  耗时: {time_cache:.2f} 秒")
    print(f"  平均: {time_cache / num_samples:.2f} 秒/样本")
    print(f"  加速: {time_no_cache / time_cache:.2f}x")

    print("\n✓ 速度测试完成!")


def main():
    parser = argparse.ArgumentParser(description="测试 RAVDESS 数据集加载")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/1188976',
        help='数据集目录'
    )
    parser.add_argument(
        '--actor',
        type=str,
        default='Actor_01',
        help='测试的 Actor'
    )
    parser.add_argument(
        '--test',
        type=str,
        default='all',
        choices=['single', 'actor', 'full', 'emotion', 'benchmark', 'all'],
        help='测试类型'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=10,
        help='最大样本数'
    )

    args = parser.parse_args()

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        return

    print(f"数据目录: {args.data_dir}")

    try:
        if args.test == 'single' or args.test == 'all':
            test_single_video(args.data_dir, args.actor)

        if args.test == 'actor' or args.test == 'all':
            test_actor_dataset(args.data_dir, args.actor, args.max_samples)

        if args.test == 'full' or args.test == 'all':
            test_full_dataset(args.data_dir, max_samples=50)

        if args.test == 'emotion' or args.test == 'all':
            test_emotion_filtering(args.data_dir, ['happy', 'sad', 'angry'])

        if args.test == 'benchmark' or args.test == 'all':
            benchmark_loading_speed(args.data_dir, num_samples=5)

        print("\n" + "=" * 60)
        print("所有测试完成! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
