#!/bin/bash
# 测试 checkpoint-20000 的脚本

CHECKPOINT_PATH="/home/ssd/zml/15/Isaac-GR00T/checkpoints/libero_spatial/run_20251229_151837/checkpoint-20000"
DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
EMBODIMENT_TAG="new_embodiment"
PORT=5555

echo "=========================================="
echo "测试 checkpoint-20000"
echo "=========================================="
echo "检查点路径: $CHECKPOINT_PATH"
echo "数据配置: $DATA_CONFIG"
echo "Embodiment Tag: $EMBODIMENT_TAG"
echo ""

# 方式1: 启动推理服务（在一个终端运行）
echo "步骤1: 启动推理服务"
echo "运行以下命令（在单独的终端中）:"
echo ""
echo "python scripts/inference_service.py --server \\"
echo "    --model-path $CHECKPOINT_PATH \\"
echo "    --data-config $DATA_CONFIG \\"
echo "    --embodiment-tag $EMBODIMENT_TAG \\"
echo "    --port $PORT"
echo ""
echo "等待服务启动后，按回车继续..."
read

# 方式2: 运行评估（在另一个终端运行）
echo ""
echo "步骤2: 运行 Libero 评估"
echo "运行以下命令（在另一个终端中）:"
echo ""
echo "python examples/Libero/eval/run_libero_eval.py \\"
echo "    --task-suite-name libero_spatial \\"
echo "    --num-trials-per-task 5 \\"
echo "    --port $PORT \\"
echo "    --headless True"
echo ""

