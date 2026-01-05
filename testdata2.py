#source $(conda info --base)/etc/profile.d/conda.sh && conda activate n15 && cd /home/ssd/zml/15/Isaac-GR00T && python -c "
import torch
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.data.schema import EmbodimentTag

# 加载数据配置
data_config_cls = load_data_config('examples.Libero.custom_data_config:LiberoDataConfig')
modality_configs = data_config_cls.modality_config()
transforms = data_config_cls.transform()

# 创建数据集
dataset = LeRobotSingleDataset(
    dataset_path='/home/ssd/zml/15/Isaac-GR00T/data_spatial_converted',
    modality_configs=modality_configs,
    transforms=transforms,
    embodiment_tag=EmbodimentTag('new_embodiment'),
    video_backend='torchvision_av',
)

print('=' * 60)
print('数据集加载成功，共 {} 个样本'.format(len(dataset)))
print('=' * 60)

# 获取前3个样本
for i in range(3):
    sample = dataset[i]
    print(f'\\n===== 样本 {i} =====')
    for key, value in sample.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            print(f'{key}:')
            print(f'  dtype: {value.dtype}')
            print(f'  shape: {value.shape}')
            if value.numel() < 100 if isinstance(value, torch.Tensor) else value.size < 100:
                print(f'  value: {value}')
            else:
                print(f'  value (first 10): {value.flatten()[:10]}')
        else:
            print(f'{key}: {type(value).__name__} = {value}')

