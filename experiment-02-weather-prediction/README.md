# 实验二：气象数据温度预测

基于甘肃省三个气象站（敦煌、金昌、酒泉）的历史气象数据进行时间序列预测任务。

## 数据集

| 城市 | 文件名 | 站点 | 记录数 |
|------|--------|------|--------|
| 敦煌 | 敦煌.xls | 吉迈镇 | ~60,000 |
| 金昌 | 金昌.xls | 永昌 | ~60,000 |
| 酒泉 | 酒泉.xls | 鼎新 | ~60,000 |

**时间范围**：2005.02 - 2025.12（约 20 年，3 小时间隔记录）

### 主要气象参数

| 参数 | 含义 |
|------|------|
| T | 气温 (℃) |
| Po | 本站气压 (hPa) |
| P | 海平面气压 (hPa) |
| U | 相对湿度 (%) |
| Ff | 风速 (m/s) |
| Tx/Tn | 最高/最低气温 |
| Td | 露点温度 |
| VV | 能见度 |

## 项目结构

```
experiment-02-weather-prediction/
├── data/                    # 原始气象数据 (.xls, gitignored)
├── notebooks/               # 数据探索与可视化
├── src/                     # 建模与训练脚本
│   ├── preprocess.py        # 数据清洗、特征工程、时间切分
│   ├── train_xgboost.py     # XGBoost 基线模型
│   └── train_lstm.py        # LSTM 深度学习模型
├── models/                  # 保存的模型权重
│   ├── xgboost_T.json
│   └── lstm_T_best.pth
├── outputs/                 # 预处理后的 CSV、结果与训练日志
│   ├── train_T.csv / val_T.csv / test_T.csv
│   ├── results_xgboost_T.json
│   ├── results_lstm_T.json
│   ├── training_log_xgboost_T_*.txt
│   └── training_log_lstm_T_*.txt
├── reports/
│   ├── experiment_proposal.md   # 问题规划与方向分析
│   ├── experiment_report.md     # 实验结果报告
│   └── course_design.md         # 课程设计完整文档
├── requirements.txt
└── README.md
```

## 实验结果

### 核心指标对比

| 指标 | XGBoost | LSTM |
|------|---------|------|
| **Test R²** | **0.9949** | 0.9477 |
| Test RMSE | **0.8889** | 2.8549 |
| Test MAE | **0.6630** | 2.2640 |
| Train R² | 0.9977 | — |
| Val R² | 0.9947 | 0.9533 |
| 训练耗时 | ~3 分钟 | ~10 分钟 |
| 早停 Epoch | 499/500 | 19/50 |

**结论**：XGBoost 在强自回归特性的温度预测任务上显著优于 LSTM，Test R² 达到 0.9949，RMSE 仅 0.89°C。

### 特征重要性 Top 5（XGBoost）

| 排名 | 特征 | 重要性 | 含义 |
|------|------|--------|------|
| 1 | `T_lag8` | 31197.8 | 24 小时前的温度（日周期） |
| 2 | `T_lag1` | 14063.0 | 3 小时前的温度（短时自回归） |
| 3 | `Tx` | 2664.1 | 当日最高气温 |
| 4 | `hour_cos` | 2613.9 | 日周期余弦编码 |
| 5 | `T_lag24` | 2163.2 | 72 小时前的温度 |

## 环境依赖

```bash
pip install pandas numpy matplotlib scikit-learn xgboost torch openpyxl
```

完整依赖见 `requirements.txt`。

## 执行流程

```bash
# 1. 数据预处理（构造 33 维特征、时间切分）
python src/preprocess.py --target T --city all

# 2. XGBoost 训练（~3 分钟）
python src/train_xgboost.py --target T --n-estimators 500

# 3. LSTM 训练（~10 分钟，需 CUDA）
python src/train_lstm.py --target T --seq-len 24 --epochs 50
```

训练日志自动保存至 `outputs/training_log_*.txt`。

## 文档说明

| 文档 | 内容 |
|------|------|
| `reports/experiment_proposal.md` | 数据质量评估、6 个候选方向对比、推荐方案 |
| `reports/experiment_report.md` | 实验过程、结果汇总、特征重要性分析 |
| `reports/course_design.md` | **课程设计完整文档**：需求分析、方案设计、详细设计、系统实现、测试分析、总结反思 |

> 课程设计文档包含完整的训练日志逐 Epoch 分析、收敛过程诊断与两模型对比。
