Eagle 2.5 VL 是一套典型的视觉语言“大脑”，整体结构可以拆成三块并用若干连接件粘合在一起：

视觉分支：基础是 SiglipVisionModel（ViT 变体），把输入图像切成 patch 后编码成视觉 token。可选的 pixel shuffle/downsample 操作负责把高分辨率特征压缩到更少的 token。
语言分支：支持 LLaMA / Qwen2 / Qwen3 等 Causal LM（源码中通过 config.language_model_type 切换，对应 LlamaForCausalLM、Qwen2ForCausalLM 等）。语言模型负责处理文本、生成动作指令等。
跨模态连接器：视觉 token 经 MLP (mlp1) 映射到语言模型的 hidden size，并在语言输入序列里用特殊的 <image> token 插入；可选的 RADIOModel 额外对语言/视觉做对齐或下游任务强化。
在推理或训练时，流程大致是：图像 → 视觉编码 → MLP 对齐 → 插到文本 token 序列 → 语言模型继续处理／生成。配置和模型定义都在 gr00t/model/backbone/eagle2_hg_model 目录下：

configuration_eagle2_5_vl.py 描述各子模块尺寸、layer 数、是否开启 pixel shuffle、选取哪一层视觉特征等。
modeling_eagle2_5_vl.py 定义了 Eagle2_5_VLForConditionalGeneration 等核心类，包括视觉/语言 forward、特征抽取、LoRA 接入等。
image_processing_eagle2*.py、processing_eagle2_5_vl.py 负责把原始图像和文本包装成模型的输入。
所以可以把 Eagle 看成“Siglip 视觉塔 + LLaMA/Qwen 语言塔 + MLP 适配层”，通过配置灵活调整层数、视觉 token 下采样方式以及是否挂接额外模块。