# 实验一：苹果品质检测 —— 实验记录

> 本文件记录各次实验的配置、结果与调参决策过程。

---

## 实验 1：ResNet18 Baseline（已终止）

### 配置

| 配置项 | 值 |
|--------|-----|
| 模型 | ResNet18 (pretrained) |
| 输入尺寸 | 224×224 |
| Epochs | 50（提前终止于 Epoch 22）|
| Batch Size | 8 |
| 学习率 | 0.001 |
| 优化器 | AdamW (weight_decay=1e-4) |
| 学习率调度 | CosineAnnealingLR |
| 验证集比例 | 0.2 (30 张) |
| 随机种子 | 42 |
| 数据增强 | Resize + Crop + Flip + Rotation + ColorJitter |
| 损失函数 | CrossEntropyLoss（无权重）|

### 终止前结果（Epoch 22）

| 指标 | 值 | 状态 |
|------|-----|------|
| Train Acc | 99.17% | ⚠️ 过高 |
| Val Acc | 73.33% | ❌ 不达标 |
| **Train-Val Gap** | **25.84%** | ❌ 严重过拟合 |
| Train Loss | 0.0272 | 很低 |
| Val Loss | 1.1968 | 很高 |

### 问题诊断

1. **严重过拟合**：模型参数量（~11.7M）远大于样本量（150 张），模型记住了训练图
2. **类别不均衡影响**：diseased（14 张）和 cracked（16 张）样本少，模型倾向预测大类
3. **正则化不足**：weight_decay=1e-4 过弱，无法有效约束模型复杂度

### 决策：终止 Baseline，切换模型并加强正则化

---

## 实验 2：EfficientNet-B0 + 优化策略（进行中）

### 配置变更

| 配置项 | 实验 1 | 实验 2 | 变更理由 |
|--------|--------|--------|----------|
| 模型 | ResNet18 | **EfficientNet-B0** | 参数量更少（~5.3M），过拟合风险更低；ImageNet Top-1 更高（77.38%）|
| Dropout | 无 | **0.5** | 强制模型更鲁棒，抑制过拟合 |
| 类别权重 | 无 | **有** | diseased/cracked 样本少，提高其权重防止被忽视 |
| weight_decay | 1e-4 | **5e-4** | 更强的 L2 正则化，惩罚大权重 |
| 其他 | 不变 | 不变 | lr=0.001, batch=8, epochs=50, seed=42 |

### 类别权重设计

| 类别 | 样本数 | 权重 | 理由 |
|------|--------|------|------|
| black_spot | 20 | 1.0 | 标准权重 |
| bruised | 20 | 1.0 | 标准权重 |
| cracked | 16 | **1.25** | 样本偏少，提升权重 |
| diseased | 14 | **1.4** | 样本最少，最高权重 |
| fresh | 20 | 1.0 | 标准权重 |
| insect_damaged | 20 | 1.0 | 标准权重 |
| rotten | 20 | 1.0 | 标准权重 |
| wrinkled | 20 | 1.0 | 标准权重 |

### 预期效果

| 指标 | 实验 1 | 预期实验 2 | 提升来源 |
|------|--------|-----------|----------|
| Val Acc | 73.33% | **88-93%** | EfficientNet + Dropout + 类别权重 + 强正则化 |
| Train-Val Gap | 25.84% | **< 15%** | Dropout + weight_decay 抑制过拟合 |
| diseased Recall | 低（推测）| **提升** | 类别权重直接提升小类识别率 |
| cracked Recall | 低（推测）| **提升** | 类别权重直接提升小类识别率 |

### 训练命令

```bash
cd "C:/Users/22414/dev/ml-course-experiments/experiment-01-apple-detection/src"
python train.py --epochs 50 --batch-size 8 --lr 0.001 --val-split 0.2 --seed 42
```

---

## 实验记录规范

每次实验完成后，在此追加记录：

```markdown
## 实验 N：[描述]

### 配置
[配置表格]

### 结果
[指标表格]

### 分析
[与上次实验对比，分析变化原因]

### 下一步决策
[继续优化 / 已达到目标 / 换方向]
```
