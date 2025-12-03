# 🌸 Fine-Grained Flower Classification with DINOv2

[](https://pytorch.org/)
[](https://github.com/facebookresearch/dinov2)
[](https://www.google.com/search?q=)

这是一个基于 **DINOv2 (Vision Transformer)** 的高精度花卉图像分类解决方案。项目针对细粒度分类任务（152类）进行了深度优化，采用了 **三阶段渐进式微调（Progressive Fine-Tuning）** 策略，在保证极高准确率（\>97.5%）的同时，满足模型体积（\<500MB）和推理速度（\<100ms）的严格工程限制。

## ✨ 核心特性

  * **SOTA 架构**: 采用 Meta 的 `DINOv2` (ViT-Base) 作为特征提取器。
  * **渐进式训练**: 自动执行 `Head` -\> `Deep Layers` -\> `Full Fine-Tuning` 三阶段训练，防止灾难性遗忘。
  * **鲁棒工程**:
      * 自动处理损坏/过小图像，防止训练中断 (NaN Loss 保护)。
      * 支持动态分辨率输入 (336px / 448px / 518px)。
      * 支持断点续训 (`--resume`)。
  * **极速推理**: 集成 FP16 半精度推理与 TTA (Test-Time Augmentation) 加速。
  * **模型压缩**: 提供脚本将模型压缩至 FP16 格式，体积减小 50%。

## 📂 项目结构

此仓库仅包含核心代码文件，建议在同级目录下创建 `data/` 和 `model/` 目录用于存放数据和输出。

```text
.
├── compress_model.py   # 模型压缩工具 (Strip Optimizer + FP16 Quantization)
├── fix_config.py       # 配置文件修复/生成工具
├── model.py            # DINOv2 模型定义、权重自动下载、层解冻逻辑
├── predict.py          # 高性能推理脚本 (支持 TTA 和 FP16)
├── requirements.txt    # 项目依赖
├── score.py            # 结果评分与错误分析工具
├── train.py            # 主训练脚本 (包含三阶段微调逻辑)
└── utils.py            # 数据加载器、增强策略、辅助函数
```

## 🚀 快速开始

### 1\. 环境安装

```bash
pip install -r requirements.txt
```

### 2\. 模型训练

训练脚本会自动下载 DINOv2 预训练权重。建议使用 **336px** 或 **448px** 分辨率以平衡速度与精度。

```bash
# 示例：使用 336px 分辨率，自动划分 20% 验证集，启用 TTA 验证
python3 train.py \
    --train_csv ./data/train_labels.csv \
    --train_img_dir ./data/train_images \
    --save_dir ./output_model \
    --model_type vit_base_patch14_dinov2.lvd142m \
    --img_size 336 \
    --batch_size 24 \
    --epochs 25 \
    --val_split 0.2 \
    --val_tta \
    --label_smoothing 0.1
```

**训练阶段说明：**

  * **Epoch 1-3**: 只训练分类头 (LR=1e-3)。
  * **Epoch 4-8**: 解冻 Transformer 后 4 层 (LR=1e-4)。
  * **Epoch 9+**: 全参数微调 (差分学习率, Backbone=5e-5, Head=5e-4)。

### 3\. 模型压缩 (可选但推荐)

为了满足 \<500MB 的限制并加速推理，建议在训练结束后压缩模型。

```bash
# 将 500MB+ 的模型压缩至 ~166MB
python3 compress_model.py ./output_model/best_model.pth ./output_model/best_model_compressed.pth
```

### 4\. 推理预测

支持两种模式：标准模式（极速）和 TTA 模式（高精）。

```bash
# 模式 A: 标准推理 (速度最快，<50ms)
python3 predict.py ./data/test_images results.csv

# 模式 B: TTA 增强推理 (精度更高，5-Crop)
python3 predict.py ./data/test_images results_tta.csv --use_tta --tta_level 5
```

### 5\. 结果评估

对比预测结果与真实标签，计算准确率并导出正确/错误样本。

```bash
python3 score.py results.csv ./data/val_labels.csv --output correct_samples.csv
```

## 🔧 高级功能

  * **断点续训**: 训练意外中断？使用 `--resume` 参数：
    ```bash
    python3 train.py ... --resume ./output_model/latest_checkpoint.pth
    ```
  * **手动权重**: 如果无法联网，请将 `vit_base_patch14_dinov2.lvd142m.pth` 放入 `weights/` 目录，代码会自动加载。
  * **修复配置**: 如果 `config.json` 损坏，可运行：
    ```bash
    python3 fix_config.py --model_dir ./output_model --img_size 336
    ```

## 📋 技术细节

  * **Backbone**: `vit_base_patch14_dinov2.lvd142m`
  * **Head**: Custom MLP (Linear -\> BatchNorm -\> ReLU -\> Dropout -\> Linear)
  * **Optimizer**: AdamW
  * **Scheduler**: CosineAnnealingLR (仅在 Stage 3 启用)
  * **Augmentation**: RandomResizedCrop, Flip, ColorJitter, **RandomErasing**