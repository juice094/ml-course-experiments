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
5. **[课程要求]** 在验证集上达到指定准确率目标，并分析影响模型性能的关键因素。

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

### 2.4 准确率要求

> **课程要求**：验证集准确率（Val Acc）需达到 **90% 以上**。
>
> **基准预期**：在 170 张小样本数据集上，未经优化的 ResNet18 验证准确率通常在 75-85% 区间。要达到 90% 以上，需要采用针对性的优化策略（详见附录 E）。

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

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Best |
|-------|-----------|-----------|----------|---------|-----|------|
| 1 | — | — | — | — | — | |
| 10 | — | — | — | — | — | |
| 20 | — | — | — | — | — | |
| 30 | — | — | — | — | — | |
| 40 | — | — | — | — | — | |
| 50 | — | — | — | — | — | |

**最优验证准确率**：`___%`（Epoch `___`）

### 6.2 准确率达标情况

> **课程要求**：验证集准确率 >= 90%

| 指标 | 要求值 | 实际值 | 是否达标 |
|------|--------|--------|----------|
| 验证集准确率（Val Acc） | >= 90.00% | `___%` | [ ] 是 / [ ] 否 |
| 训练集准确率（Train Acc） | — | `___%` | — |
| Train-Val Gap | < 15%（防止过拟合） | `___%` | [ ] 是 / [ ] 否 |

**达标分析**：

> （请填写：是否达到 90% 要求？如果未达标，分析主要原因。如果达标，说明采用了哪些关键优化策略。）

### 6.3 训练曲线

> 请在此处插入 TensorBoard 截图或 matplotlib 绘制的 Loss/Accuracy 曲线。
> 
> 建议包含：
> - 训练集与验证集 Loss 曲线
> - 训练集与验证集 Accuracy 曲线
> - 学习率变化曲线

### 6.4 混淆矩阵

> 请运行 `src/evaluate.py` 生成混淆矩阵并粘贴结果。
> 
> 重点关注：
> - diseased / cracked 样本是否被正确分类
> - 哪些类别之间容易发生混淆

### 6.5 每类性能分析

> 请从 `evaluate.py` 的输出中提取 Precision、Recall、F1-Score：

| 类别 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| fresh | — | — | — | — |
| diseased | — | — | — | — |
| bruised | — | — | — | — |
| rotten | — | — | — | — |
| insect_damaged | — | — | — | — |
| cracked | — | — | — | — |
| wrinkled | — | — | — | — |
| black_spot | — | — | — | — |
| **Overall** | — | — | — | — |

**低 Recall 类别分析**：

> （请填写：哪些类别的 Recall 明显偏低？与样本量是否相关？）

### 6.6 测试集预测结果

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

### 7.1 准确率达标分析

> **核心问题：是否达到 90% 验证准确率要求？**

#### 7.1.1 达标情况

- **实际最优 Val Acc**：`___%`
- **与目标差距**：`___%`
- **是否达标**：[ ] 是 / [ ] 否

#### 7.1.2 关键优化策略（如达标）

> 列出对提升准确率最有效的 2-3 项策略，按贡献度排序：

| 优化策略 | 实施方式 | 对 Val Acc 的提升 |
|----------|----------|------------------|
| （如：类别权重） | 为 diseased/cracked 设置更高权重 | +_ _% |
| （如：冻层训练） | 先冻结 backbone 训练 10 epoch，再解冻微调 | +_ _% |
| （如：更强的增强） | 添加 RandomAffine、AutoAugment | +_ _% |

#### 7.1.3 未达标原因分析（如未达标）

> 从以下维度分析：
> 1. **数据层面**：样本量不足？类别不均衡严重？
> 2. **模型层面**：模型容量不够？欠拟合/过拟合？
> 3. **训练层面**：学习率不合适？训练不充分？
> 4. **验证划分**：是否因随机划分导致验证集过难/过易？

### 7.2 模型性能评估

- 训练准确率与验证准确率之间的差距（Train-Val Gap）是多少？
- Gap > 15% 说明过拟合，Gap < 5% 说明模型可能欠拟合或数据划分有问题。

### 7.3 类别不均衡影响

- diseased（14 张）和 cracked（16 张）的样本量是否影响了模型对这两类的识别？
- 从混淆矩阵和 per-class recall 中找出证据。

### 7.4 数据增强效果

- ColorJitter 和 RandomRotation 是否有效提升了模型的泛化能力？
- 是否尝试过移除某些增强操作进行对比实验？

### 7.5 改进方向与展望

1. **更多数据**：收集额外样本，尤其是 diseased 和 cracked 类别。
2. **模型升级**：尝试 ResNet34、EfficientNet-B0 等更深/更高效的网络。
3. **更细粒度的验证**：K-Fold 交叉验证，避免随机划分的偏差。
4. **超参数调优**：学习率（如 0.0001）、batch size、weight decay 的网格搜索。
5. **集成学习**：训练多个模型，取平均预测。
6. **测试时增强（TTA）**：推理时对同一张图多次增强取平均，提升稳定性。

---

## 八、实验总结

### 8.1 主要结论

> （请根据实际结果填写，需包含准确率达标情况的明确结论）
>
> 例如：
> 本实验基于 ResNet18 迁移学习完成了苹果品质 8 分类任务。在 170 张训练图像上训练 50 epoch 后，模型在验证集上达到了 `___%` 的准确率，**达到/未达到**课程要求的 90% 目标。实验表明……

### 8.2 遇到的问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| （待填写） | （待填写） |

### 8.3 收获与反思

> （请填写个人对本次实验的理解、收获及可改进之处）

---

## 附录

### A. 项目目录结构

```
experiment-01-apple-detection/
├── data/
│   ├── train/          # 训练数据（8 类）
│   ├── test/           # 测试数据（30 张）
│   └── raw/            # 原始数据备份
├── models/             # 保存的模型权重
├── notebooks/
│   └── exploration.ipynb   # 数据探索
├── src/
│   ├── train.py        # 训练脚本（含文本日志/JSON 导出）
│   ├── predict.py      # 推理脚本
│   └── evaluate.py     # 评估与混淆矩阵
├── outputs/
│   ├── runs/           # TensorBoard 日志
│   ├── training_log_*.txt   # 文本训练日志
│   └── metrics_*.json  # JSON 指标
├── reports/
│   ├── experiment_report.md    # 本报告
│   └── generate_plots.py     # 图表生成脚本
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

### D. 评估命令

```bash
# 验证集评估 + 混淆矩阵
python evaluate.py --model ../models/best_model.pth --val-split 0.2

# 生成报告图表
cd ../reports
python generate_plots.py
```

### E. 达到 90% 准确率的优化策略参考

> **警告**：默认配置下验证准确率预期为 75-85%。要达到 90% 以上，需组合使用以下策略。

#### E.1 策略一：类别权重（解决不均衡）

在 `train.py` 的 CrossEntropyLoss 中增加 `weight` 参数：

```python
# 计算类别权重（样本越少，权重越高）
class_counts = [20, 14, 20, 20, 20, 16, 20, 20]
total = sum(class_counts)
class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**预期提升**：+3~5%（对小样本类别的 recall 改善明显）

#### E.2 策略二：冻层训练（稳定收敛）

分阶段训练：
1. **阶段 1**（Epoch 1-10）：冻结 ResNet18 backbone 所有层，只训练新 FC 层。`lr=0.001`
2. **阶段 2**（Epoch 11-50）：解冻全部层，使用更小的学习率 `lr=0.0001` 微调。

```python
# 阶段 1：冻结 backbone
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

# 阶段 2：解冻
for param in model.parameters():
    param.requires_grad = True
```

**预期提升**：+2~4%（避免预训练特征被破坏）

#### E.3 策略三：更强的数据增强

在 `get_transforms()` 中补充：

```python
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
```

或使用 `torchvision.transforms.AutoAugment`：

```python
transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)
```

**预期提升**：+2~3%（提升泛化能力）

#### E.4 策略四：模型升级（EfficientNet-B0）

替换 ResNet18 为 EfficientNet-B0：

```python
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
```

**预期提升**：+3~5%（更优的特征提取能力）

#### E.5 策略五：K-Fold 交叉验证

使用 5-Fold 交叉验证取代单次随机划分：
- 每折训练后保存最佳模型
- 最终取 5 个模型的平均预测（集成）

**预期提升**：+2~4%（降低随机划分偏差）

#### E.6 策略组合建议

| 优先级 | 策略 | 难度 | 预期提升 |
|--------|------|------|----------|
| 高 | 类别权重 | 低 | +3~5% |
| 高 | 冻层训练 | 中 | +2~4% |
| 中 | 更强的增强 | 低 | +2~3% |
| 中 | 模型升级（EfficientNet） | 低 | +3~5% |
| 低 | K-Fold + 集成 | 高 | +2~4% |

**保守估计**：组合使用高优先级策略（类别权重 + 冻层训练），预期可达 **88-92%**。若再加上 EfficientNet，可达 **90-95%**。

#### E.7 调参速查表

| 现象 | 诊断 | 调整方案 |
|------|------|----------|
| Val Acc 停滞在 80% 左右 | 模型欠拟合 | 增加 epochs 到 100，减小 lr 到 0.0005 |
| Train Acc 95%, Val Acc 75% | 严重过拟合 | 增强数据增强，增大 weight_decay 到 5e-4，加 Dropout |
| diseased/cracked recall < 50% | 类别不均衡 | 使用类别权重，或对这些类过采样 |
| Loss 震荡不收敛 | 学习率过大 | lr 减半，或改用 StepLR 替代 CosineAnnealing |

### F. 参考资源

- PyTorch 官方文档：https://pytorch.org/docs/
- Torchvision 模型库：https://pytorch.org/vision/stable/models.html
- TensorBoard 使用教程：https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
- CrossEntropyLoss 类别权重：https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
