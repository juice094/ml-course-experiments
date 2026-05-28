# 苹果品质检测机器学习实验报告

> 实验日期：2026-05-28  
> 实验环境：Python 3.x + PyTorch 2.x + CUDA/CPU  
> 实验者：（请填写）

---

## 一、实验目的

1. 掌握基于深度学习的图像分类完整流程：数据预处理、模型构建、训练、评估与推理。
2. 理解迁移学习（Transfer Learning）在小样本场景下的应用与效果。
3. 学习使用 PyTorch 框架搭建 ResNet18 分类模型，并监控训练过程。
4. 通过实际数据集，分析模型在苹果品质 8 分类任务上的性能表现。

---

## 二、实验环境与配置

### 2.1 软硬件环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 11 |
| Python 版本 | 3.x |
| PyTorch 版本 | >= 2.0.0 |
| torchvision 版本 | >= 0.15.0 |
| GPU 支持 | CUDA / CPU Fallback |
| 开发工具 | Jupyter Lab, TensorBoard |

### 2.2 关键依赖

```text
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tensorboard>=2.13.0
```

### 2.3 实验超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 模型架构 | ResNet18 | 预训练权重：ImageNet1K_V1 |
| 输入尺寸 | 224 x 224 | 标准化均值 [0.485, 0.456, 0.406]，标准差 [0.229, 0.224, 0.225] |
| Epochs | 50 | 可根据收敛情况调整 |
| Batch Size | 8 | 小批量，适应小数据集 |
| 学习率 | 0.001 | AdamW 优化器 |
| 权重衰减 | 1e-4 | L2 正则化 |
| 学习率调度 | CosineAnnealingLR | T_max = epochs |
| 验证集比例 | 0.2 | 约 34 张用于验证 |
| 随机种子 | 42 | 保证实验可复现 |

---

## 三、数据集描述

### 3.1 数据集来源

本实验使用苹果品质检测图像数据集，共包含 **8 个类别** 的苹果图像，涵盖合格品与 7 种常见缺陷类型。

### 3.2 类别分布

| 类别（英文） | 类别（中文） | 训练样本数 | 占比 |
|--------------|--------------|-----------|------|
| fresh | 合格 | 20 | ~11.8% |
| diseased | 病变 | 14 | ~8.2% |
| bruised | 碰伤 | 20 | ~11.8% |
| rotten | 腐烂 | 20 | ~11.8% |
| insect_damaged | 虫伤 | 20 | ~11.8% |
| cracked | 裂果 | 16 | ~9.4% |
| wrinkled | 褶皱 | 20 | ~11.8% |
| black_spot | 黑斑 | 20 | ~11.8% |
| **合计** | — | **170** | **100%** |

### 3.3 数据特点分析

- **样本总量小**：仅 170 张训练图像，属于典型的小样本学习场景。
- **类别不均衡**：`diseased`（14 张）和 `cracked`（16 张）样本偏少，可能影响模型对这两类的识别能力。
- **测试集规模**：30 张无标签图像，用于最终推理预测。

### 3.4 数据增强策略

为缓解小样本带来的过拟合风险，训练阶段采用以下增强：

- Resize(256) + RandomCrop(224)
- RandomHorizontalFlip(p=0.5)
- RandomRotation(degrees=15)
- ColorJitter(brightness=0.2, contrast=0.2)

验证/测试阶段仅使用 Resize(224) 和标准化。

---

## 四、模型架构

### 4.1 主干网络：ResNet18

- **来源**：Torchvision 预训练模型（ImageNet1K_V1）
- **参数规模**：约 11.7M 参数
- **修改**：替换最后一层全连接层，输出维度由 1000 改为 8（对应 8 个类别）

### 4.2 模型结构简图

```
Input (3 x 224 x 224)
    |
ResNet18 Backbone (预训练)
    | Conv + BatchNorm + ReLU + MaxPool
    | Layer1 (64 channels)
    | Layer2 (128 channels)
    | Layer3 (256 channels)
    | Layer4 (512 channels)
    | Global Average Pooling
    v
FC Layer (512 -> 8)
    |
Softmax (推理阶段)
    v
Output: 8-class probabilities
```

### 4.3 选择 ResNet18 的理由

- 网络较浅（18 层），参数量适中，在 CPU 上也可较快完成训练。
- ImageNet 预训练权重提供了良好的视觉特征提取能力，适合迁移到小样本分类任务。

---

## 五、实验方法与步骤

### 5.1 数据预处理

1. 按类别目录组织训练图像（`data/train/<class>/`）。
2. 使用 `torchvision.datasets.ImageFolder` 自动加载并映射类别标签。
3. 按 8:2 比例划分为训练集和验证集（`random_split`，seed=42）。

### 5.2 训练流程

1. 加载预训练 ResNet18，替换 FC 层。
2. 定义损失函数：`CrossEntropyLoss`。
3. 定义优化器：`AdamW(lr=0.001, weight_decay=1e-4)`。
4. 定义学习率调度器：`CosineAnnealingLR(T_max=50)`。
5. 训练循环：每个 epoch 结束后在验证集上评估，保存验证准确率最高的模型。
6. 使用 TensorBoard 实时记录 loss、accuracy、learning rate。

### 5.3 推理流程

1. 加载保存的最优模型权重。
2. 对测试集图像进行预处理（Resize 224 + Normalize）。
3. 执行前向传播，输出 top-1 预测类别及置信度。
4. 批量推理结果导出为 JSON 格式。

---

## 六、实验结果

### 6.1 训练日志摘要

> 请根据实际训练结果填写。示例格式如下：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|-------|-----------|-----------|----------|---------|-----|
| 1 | — | — | — | — | — |
| 10 | — | — | — | — | — |
| 20 | — | — | — | — | — |
| 30 | — | — | — | — | — |
| 40 | — | — | — | — | — |
| 50 | — | — | — | — | — |

**最优验证准确率**：`___%`（Epoch `___`）

### 6.2 训练曲线

> 请在此处插入 TensorBoard 截图或 matplotlib 绘制的 Loss/Accuracy 曲线。
> 
> 建议包含：
> - 训练集与验证集 Loss 曲线
> - 训练集与验证集 Accuracy 曲线
> - 学习率变化曲线

### 6.3 混淆矩阵

> 请运行 `src/evaluate.py`（如有）生成混淆矩阵并粘贴结果。
> 
> 重点关注：
> - diseased / cracked 样本是否被正确分类
> - 哪些类别之间容易发生混淆

### 6.4 测试集预测结果

> 请运行推理并汇总 top-1 预测分布：

| 预测类别 | 数量 | 占比 |
|----------|------|------|
| 合格 | — | — |
| 病变 | — | — |
| 碰伤 | — | — |
| 腐烂 | — | — |
| 虫伤 | — | — |
| 裂果 | — | — |
| 褶皱 | — | — |
| 黑斑 | — | — |

---

## 七、结果分析与讨论

### 7.1 模型性能评估

- 验证准确率是否达到预期（README 预期 75-85%）？
- 训练准确率与验证准确率之间的差距是否过大？是否出现过拟合？

### 7.2 类别不均衡影响

- diseased（14 张）和 cracked（16 张）的样本量是否影响了模型对这两类的识别？
- 可考虑：过采样（oversampling）、类别权重（class weights）、或收集更多样本。

### 7.3 数据增强效果

- ColorJitter 和 RandomRotation 是否有效提升了模型的泛化能力？
- 是否尝试过移除某些增强操作进行对比实验？

### 7.4 改进方向

1. **更多数据**：收集额外样本，尤其是 diseased 和 cracked 类别。
2. **模型升级**：尝试 ResNet34、EfficientNet-B0 等更深/更高效的网络。
3. **更细粒度的验证**：K-Fold 交叉验证，避免随机划分的偏差。
4. **超参数调优**：学习率（如 0.0001）、batch size、weight decay 的网格搜索。
5. **集成学习**：训练多个模型，取平均预测。

---

## 八、实验总结

### 8.1 主要结论

（请根据实际结果填写，例如：）

> 本实验基于 ResNet18 迁移学习完成了苹果品质 8 分类任务。在 170 张训练图像上训练 50 epoch 后，模型在验证集上达到了 `___%` 的准确率。实验表明，在小样本条件下，ImageNet 预训练权重能有效提升模型收敛速度和最终性能。同时，数据增强策略对抑制过拟合起到了积极作用。

### 8.2 遇到的问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| （待填写） | （待填写） |

### 8.3 收获与反思

（请填写个人对本次实验的理解、收获及可改进之处）

---

## 附录

### A. 项目目录结构

```
ml-apple-detection/
├── data/
│   ├── train/          # 训练数据（8 类）
│   ├── test/           # 测试数据（30 张）
│   └── raw/            # 原始数据备份
├── models/             # 保存的模型权重
├── notebooks/
│   └── exploration.ipynb   # 数据探索
├── src/
│   ├── train.py        # 训练脚本
│   └── predict.py      # 推理脚本
├── outputs/
│   ├── runs/           # TensorBoard 日志
│   └── training_log.txt    # 文本训练日志
├── reports/
│   └── experiment_report.md    # 本报告
├── requirements.txt
└── README.md
```

### B. 训练命令

```bash
cd src
python train.py --epochs 50 --batch-size 8 --lr 0.001 --val-split 0.2 --seed 42
```

### C. 推理命令

```bash
# 批量推理
python predict.py ../data/test --model ../models/best_model.pth --output ../outputs/predictions.json
```

### D. 参考资源

- PyTorch 官方文档：https://pytorch.org/docs/
- Torchvision 模型库：https://pytorch.org/vision/stable/models.html
- TensorBoard 使用教程：https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
