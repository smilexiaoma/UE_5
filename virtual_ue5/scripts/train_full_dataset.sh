#!/bin/bash
# 全数据集训练脚本
# 使用所有Actors进行完整训练

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 配置参数
DATA_DIR="/root/autodl-tmp/digietal_data"
BATCH_SIZE=32  # RTX 4090 24GB - 全数据集使用更大批次提升训练效率
SEQ_LEN=100
EPOCHS=100
LR=1e-4

# 创建脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "全数据集训练配置"
echo "=========================================="
echo "数据集: 所有Actors"
echo "批次大小: $BATCH_SIZE"
echo "序列长度: $SEQ_LEN"
echo "训练轮数: $EPOCHS"
echo "预计样本数: ~4900个视频"
echo "=========================================="

# 1. Baseline Audio Model
echo ""
echo "训练 Baseline Audio Model..."
python train_ravdess.py \
    --model base_audio \
    --data_dir "$DATA_DIR" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --save_dir results/training/full_dataset \
    --use_cache \
    --num_workers 4

# 2. Baseline Video Model
echo ""
echo "训练 Baseline Video Model..."
echo "注意: Base Video 模型会自动使用优化配置防止NaN"
echo "  - 学习率自动降低到 20% (1e-4 → 2e-5)"
echo "  - 梯度裁剪更严格 (max_norm=0.5)"
python train_ravdess.py \
    --model base_video \
    --data_dir "$DATA_DIR" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --save_dir results/training/full_dataset \
    --use_cache \
    --num_workers 4

# 3. Dual Fusion Model (创新点1)
echo ""
echo "训练 Dual Fusion Model (创新点1)..."
python train_ravdess.py \
    --model dual_fusion \
    --data_dir "$DATA_DIR" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --save_dir results/training/full_dataset \
    --use_cache \
    --num_workers 4

# 4. Diffusion Model (创新点2)
echo ""
echo "训练 Diffusion Model (创新点2)..."
python train_ravdess.py \
    --model diffusion \
    --data_dir "$DATA_DIR" \
    --batch_size 16 \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --save_dir results/training/full_dataset \
    --use_cache \
    --num_workers 4

# 5. E2E Loop Model (创新点3)
echo ""
echo "训练 E2E Loop Model (创新点3)..."
python train_ravdess.py \
    --model e2e_loop \
    --data_dir "$DATA_DIR" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --save_dir results/training/full_dataset \
    --use_cache \
    --num_workers 4

echo ""
echo "=========================================="
echo "所有全数据集实验完成!"
echo "=========================================="
