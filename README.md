# 机器学习实验仓库

本仓库用于存放课程要求的机器学习实验项目代码、报告与说明文档。

## 实验列表

| 实验编号 | 名称 | 类型 | 状态 |
|---------|------|------|------|
| 实验一 | [苹果品质检测](./experiment-01-apple-detection/) | 图像分类 (8-class) | 代码就绪，待训练 |
| 实验二 | [气象数据预测](./experiment-02-weather-prediction/) | 时间序列预测 | 数据就绪，待建模 |

## 仓库结构

```
ml-course-experiments/
├── experiment-01-apple-detection/    # 苹果品质检测
│   ├── data/                          # 数据集目录 (gitignored)
│   ├── models/                        # 模型权重 (gitignored)
│   ├── notebooks/                     # 数据探索 notebook
│   ├── src/                           # 训练/推理/评估脚本
│   ├── reports/                       # 实验报告与图表
│   ├── outputs/                       # 训练日志与预测结果
│   └── requirements.txt               # Python 依赖
│
├── experiment-02-weather-prediction/  # 气象数据预测
│   ├── data/                          # 原始气象数据 (gitignored)
│   ├── notebooks/                     # 数据探索与分析
│   ├── src/                           # 建模脚本
│   ├── reports/                       # 实验报告
│   └── requirements.txt               # Python 依赖
│
├── .gitignore                         # 全局忽略规则
└── README.md                          # 本文件
```

## 注意事项

- **数据集不纳入版本控制**：图片和原始 Excel 数据文件体积过大，已加入 `.gitignore`。
- 各实验的数据集需从课程提供的位置获取后，放置到对应 `data/` 目录下。
- 实验报告模板位于各实验的 `reports/` 目录中。
