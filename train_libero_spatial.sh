#!/bin/bash
# 训练脚本：使用转换后的 Libero spatial 数据集训练 GR00T
# 参数与官方 examples/Libero/README.md 保持一致

# 激活 conda 环境

# 使用 2 张 GPU
echo "使用 2 张 GPU 进行训练"

# 抑制警告信息
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"

# 检查环境
echo "检查环境..."
python -c "import torch; import numpy; print(f'PyTorch: {torch.__version__}, NumPy: {numpy.__version__}')" || {
    echo "环境检查失败，请确保已激活正确的 conda 环境"
    exit 1
}

# 数据集路径
DATASET_DIR="/home/ssd/zml/15/Isaac-GR00T/data_spatial_converted"

# 检查数据集是否存在
if [ ! -d "$DATASET_DIR" ] || [ ! -f "$DATASET_DIR/meta/episodes.jsonl" ]; then
    echo "错误: 转换后的数据集不存在"
    echo "请先运行转换脚本:"
    echo "  python scripts/convert_libero_native_to_training_format.py \\"
    echo "      --input_dir /home/ssd/zml/15/Isaac-GR00T/data_spatial \\"
    echo "      --output_dir $DATASET_DIR"
    exit 1
fi

# 检查数据集完整性
VIDEO_COUNT=$(find "$DATASET_DIR/videos" -name "*.mp4" 2>/dev/null | wc -l)
EPISODES_COUNT=$(wc -l < "$DATASET_DIR/meta/episodes.jsonl" 2>/dev/null || echo "0")
echo "数据集检查: 视频文件数量=$VIDEO_COUNT, Episodes数量=$EPISODES_COUNT"

# Checkpoint 保存目录
PROJECT_ROOT="/home/ssd/zml/15/Isaac-GR00T"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/libero_spatial"
OUTPUT_DIR="$CHECKPOINT_DIR/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECKPOINT_DIR"

# 训练配置（与官方 README 保持一致）
# 官方: max_steps=20K, batch_size=128, grad_accum_steps=1, num_gpus=8
BATCH_SIZE=128
GRAD_ACCUM_STEPS=4
NUM_GPUS=2
MAX_STEPS=20000
SAVE_STEPS=2000
LEARNING_RATE=0.0001  # 学习率，默认 1e-4

echo "开始训练..."
echo "数据集路径: $DATASET_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "配置: max_steps=$MAX_STEPS, batch_size=$BATCH_SIZE, grad_accum_steps=$GRAD_ACCUM_STEPS, num_gpus=$NUM_GPUS"
echo "等效 batch size: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))"

# 运行训练脚本
python scripts/gr00t_finetune.py \
    --dataset-path "$DATASET_DIR" \
    --data_config examples.Libero.custom_data_config:LiberoDataConfig \
    --num-gpus $NUM_GPUS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM_STEPS \
    --max-steps $MAX_STEPS \
    --output-dir "$OUTPUT_DIR" \
    --save-steps $SAVE_STEPS \
    --learning-rate $LEARNING_RATE \
    --embodiment-tag new_embodiment \
    --video-backend torchvision_av

if [ $? -eq 0 ]; then
    echo "训练完成！检查点保存在: $OUTPUT_DIR"
else
    echo "训练过程中出现错误，请检查上面的错误信息"
    exit 1
fi
