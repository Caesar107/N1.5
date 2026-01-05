#!/usr/bin/env python
"""
测试脚本：检查输入到 VLM 和 Action Head 的数据格式和内容
"""

import sys
import torch
import numpy as np
from datetime import datetime

# 输出重定向到文件
output_file = "/home/ssd/zml/15/Isaac-GR00T/test_data_input_output.txt"

class Tee:
    """同时输出到控制台和文件"""
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()

# 开始记录
tee = Tee(output_file)
sys.stdout = tee

print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

try:
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config
    from gr00t.data.schema import EmbodimentTag
    from gr00t.model.gr00t_n1 import GR00T_N1_5

    # ========== 1. 加载数据配置 ==========
    print("\n[1] 加载数据配置...")
    data_config_cls = load_data_config('examples.Libero.custom_data_config:LiberoDataConfig')
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    
    print(f"video_keys: {data_config_cls.video_keys}")
    print(f"state_keys: {data_config_cls.state_keys}")
    print(f"action_keys: {data_config_cls.action_keys}")
    print(f"language_keys: {data_config_cls.language_keys}")
    print(f"observation_indices: {data_config_cls.observation_indices}")
    print(f"action_indices: {data_config_cls.action_indices}")

    # ========== 2. 创建数据集 ==========
    print("\n[2] 创建数据集...")
    dataset = LeRobotSingleDataset(
        dataset_path='/home/ssd/zml/15/Isaac-GR00T/data_spatial_converted',
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=EmbodimentTag('new_embodiment'),
        video_backend='torchvision_av',
    )
    print(f"数据集大小: {len(dataset)} 个样本")

    # ========== 3. 检查前3个样本 ==========
    print("\n" + "=" * 80)
    print("[3] 检查前3个样本的数据格式")
    print("=" * 80)

    for sample_idx in range(3):
        sample = dataset[sample_idx]
        print(f"\n{'='*30} 样本 {sample_idx} {'='*30}")
        
        # ----- 输入 VLM 的数据 (eagle_content) -----
        print("\n>>> 输入 VLM 的数据 (eagle_content):")
        eagle_content = sample['eagle_content']
        print(f"  image_inputs: {len(eagle_content['image_inputs'])} 张图像")
        for i, img in enumerate(eagle_content['image_inputs']):
            print(f"    image_{i}: size={img.size}, mode={img.mode}")
        print(f"  video_inputs: {eagle_content['video_inputs']}")
        print(f"  text_list:")
        for txt in eagle_content['text_list']:
            print(f"    {repr(txt[:200])}..." if len(txt) > 200 else f"    {repr(txt)}")
        
        # ----- State -----
        print("\n>>> State (输入到模型的状态):")
        state = sample['state']
        state_mask = sample['state_mask']
        print(f"  shape: {state.shape}")
        print(f"  dtype: {state.dtype}")
        print(f"  有效维度 (mask=True): {state_mask.sum()}")
        print(f"  有效值 (前8个):")
        valid_state = state[0, :8]
        print(f"    x={valid_state[0]:.6f}, y={valid_state[1]:.6f}, z={valid_state[2]:.6f}")
        print(f"    roll={valid_state[3]:.6f}, pitch={valid_state[4]:.6f}, yaw={valid_state[5]:.6f}")
        print(f"    gripper=[{valid_state[6]:.6f}, {valid_state[7]:.6f}]")
        
        # ----- Action -----
        print("\n>>> Action (训练目标 / Action Head 输出目标):")
        action = sample['action']
        action_mask = sample['action_mask']
        print(f"  shape: {action.shape}  (action_horizon={action.shape[0]}, max_action_dim={action.shape[1]})")
        print(f"  dtype: {action.dtype}")
        print(f"  每步有效维度 (mask=True): {action_mask[0].sum()}")
        print(f"  前3个时间步的有效值 (前7个):")
        for t in range(min(3, action.shape[0])):
            valid_action = action[t, :7]
            print(f"    t={t}: x={valid_action[0]:.4f}, y={valid_action[1]:.4f}, z={valid_action[2]:.4f}, "
                  f"roll={valid_action[3]:.4f}, pitch={valid_action[4]:.4f}, yaw={valid_action[5]:.4f}, "
                  f"gripper={valid_action[6]:.4f}")
        
        # ----- Embodiment ID -----
        print(f"\n>>> Embodiment ID: {sample['embodiment_id']}")

    # ========== 4. 加载模型检查 ==========
    print("\n" + "=" * 80)
    print("[4] 加载模型检查配置")
    print("=" * 80)
    
    print("\n正在加载模型 nvidia/GR00T-N1.5-3B ...")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path='nvidia/GR00T-N1.5-3B',
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
    )
    model.eval()
    
    print(f"\n>>> 模型配置:")
    print(f"  action_dim: {model.config.action_dim}")
    print(f"  action_horizon: {model.config.action_horizon}")
    print(f"  compute_dtype: {model.config.compute_dtype}")
    
    print(f"\n>>> Action Head 配置:")
    print(f"  action_dim: {model.action_head.config.action_dim}")
    print(f"  action_horizon: {model.action_head.config.action_horizon}")
    
    # ========== 5. 对比分析 ==========
    print("\n" + "=" * 80)
    print("[5] 对比分析")
    print("=" * 80)
    
    data_action_dim = 7  # 从 action_mask 计算得到
    data_action_horizon = len(data_config_cls.action_indices)
    model_action_dim = model.action_head.config.action_dim
    model_action_horizon = model.action_head.config.action_horizon
    
    print(f"\n>>> 数据 vs 模型:")
    print(f"  action_dim:     数据={data_action_dim}, 模型={model_action_dim}")
    print(f"  action_horizon: 数据={data_action_horizon}, 模型={model_action_horizon}")
    
    if data_action_horizon != model_action_horizon:
        print(f"\n  ⚠️ action_horizon 不匹配! 训练时会自动调整 action head")
    else:
        print(f"\n  ✅ action_horizon 匹配")
    
    if data_action_dim != model_action_dim:
        print(f"  ⚠️ action_dim 不匹配! 训练时会自动调整 action head")
    else:
        print(f"  ✅ action_dim 匹配")

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

except Exception as e:
    import traceback
    print(f"\n错误: {e}")
    traceback.print_exc()

finally:
    sys.stdout = tee.stdout
    tee.close()
    print(f"\n输出已保存到: {output_file}")

