#!/bin/bash
# Diffusion 模型结果清理脚本
# 用于清理旧版本和重复的训练结果

echo "=========================================="
echo "Diffusion 模型结果清理工具"
echo "=========================================="

# 1. 删除 improved_diffusion 目录（已复制到 innovation_training）
echo ""
echo "选项 1: 删除 results/improved_diffusion/ 目录"
echo "  该目录已复制到 results/innovation_training/"
echo "  可节省空间: ~985M"
read -p "  是否删除? (y/n): " confirm1
if [ "$confirm1" = "y" ]; then
    rm -rf results/improved_diffusion/
    echo "  ✓ 已删除 improved_diffusion 目录"
else
    echo "  - 跳过"
fi

# 2. 删除旧版本 diffusion 结果备份
echo ""
echo "选项 2: 删除旧版本 diffusion 结果 (*_old)"
echo "  diffusion_Actor_01_20251214_003034_old (~493M)"
echo "  diffusion_all_actors_20251214_003139_old (~353M)"
echo "  可节省空间: ~846M"
read -p "  是否删除? (y/n): " confirm2
if [ "$confirm2" = "y" ]; then
    rm -rf results/innovation_training/diffusion_Actor_01_20251214_003034_old
    rm -rf results/innovation_training/diffusion_all_actors_20251214_003139_old
    echo "  ✓ 已删除旧版本备份"
else
    echo "  - 跳过"
fi

# 3. 压缩旧版本备份（替代删除）
echo ""
echo "选项 3: 压缩旧版本备份为 .tar.gz（如果未删除）"
read -p "  是否压缩? (y/n): " confirm3
if [ "$confirm3" = "y" ] && [ -d "results/innovation_training/diffusion_Actor_01_20251214_003034_old" ]; then
    cd results/innovation_training/
    tar -czf diffusion_old_backups_20251214.tar.gz diffusion_*_old
    rm -rf diffusion_*_old
    cd ../..
    echo "  ✓ 已压缩为 diffusion_old_backups_20251214.tar.gz"
else
    echo "  - 跳过"
fi

echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo ""
echo "当前 diffusion 结果目录："
ls -lh results/innovation_training/ | grep diffusion
