#!/bin/bash
# 单个Actor (Actor_01 2) 训练脚本
# 用于验证实验的完整性和快速原型测试

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 配置参数
ACTOR="Actor_01"
DATA_DIR="/root/autodl-tmp/digietal_data"
BATCH_SIZE=16  # RTX 4090 24GB - 可以使用更大的批次
SEQ_LEN=100
EPOCHS=100
LR=1e-4

# 创建脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "单个Actor训练配置"
echo "=========================================="
echo "Actor: $ACTOR"
echo "批次大小: $BATCH_SIZE"
echo "序列长度: $SEQ_LEN"
echo "训练轮数: $EPOCHS"
echo "=========================================="

# 1. Baseline Audio Model
echo ""
echo "训练 Baseline Audio Model..."
python train_ravdess.py \
    --model base_audio \
    --data_dir "$DATA_DIR" \
    --actor "$ACTOR" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --save_dir results/training/single_actor \
    --use_cache

# 2. Baseline Video Model
echo ""
echo "训练 Baseline Video Model..."
echo "注意: Base Video 模型会自动使用优化配置防止NaN"
echo "  - 学习率自动降低到 20% (1e-4 → 2e-5)"
echo "  - 梯度裁剪更严格 (max_norm=0.5)"
python train_ravdess.py \
    --model base_video \
    --data_dir "$DATA_DIR" \
    --actor "$ACTOR" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --save_dir results/training/single_actor \
    --use_cache

# 3. Dual Fusion Model (创新点1)
echo ""
echo "训练 Dual Fusion Model (创新点1)..."
python train_ravdess.py \
    --model dual_fusion \
    --data_dir "$DATA_DIR" \
    --actor "$ACTOR" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --save_dir results/training/single_actor \
    --use_cache

# 4. Diffusion Model (创新点2)
echo ""
echo "训练 Diffusion Model (创新点2)..."
python train_ravdess.py \
    --model diffusion \
    --data_dir "$DATA_DIR" \
    --actor "$ACTOR" \
    --batch_size 8 \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --save_dir results/training/single_actor \
    --use_cache

# 5. E2E Loop Model (创新点3)
echo ""
echo "训练 E2E Loop Model (创新点3)..."
python train_ravdess.py \
    --model e2e_loop \
    --data_dir "$DATA_DIR" \
    --actor "$ACTOR" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --save_dir results/training/single_actor \
    --use_cache

echo ""
echo "=========================================="
echo "所有单Actor实验完成!"
echo "=========================================="
