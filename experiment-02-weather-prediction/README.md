# 实验二：气象数据预测

基于甘肃省三个气象站（敦煌、金昌、酒泉）的历史气象数据进行时间序列预测任务。

## 数据集

| 城市 | 文件名 | 站点 | WMO ID | 记录数 |
|------|--------|------|--------|--------|
| 敦煌 | 敦煌.xls | 吉迈镇 | 56046 | ~60,000 |
| 金昌 | 金昌.xls | 永昌 | 52674 | ~60,000 |
| 酒泉 | 酒泉.xls | 鼎新 | 52378 | ~60,000 |

**时间范围**：2005.02 - 2025.12（约 20 年，3 小时间隔记录）

### 主要气象参数

| 参数 | 含义 |
|------|------|
| T | 气温 (℃) |
| Po | 本站气压 (hPa) |
| P | 海平面气压 (hPa) |
| Pa | 气压变化 (hPa) |
| U | 相对湿度 (%) |
| DD | 风向 |
| Ff | 风速 (m/s) |
| ff10 | 10 分钟最大风速 |
| ff3 | 3 秒阵风风速 |
| N | 总云量 |
| WW | 现在天气现象 |
| W1/W2 | 过去天气现象 |
| Tn/Tx | 最低/最高气温 |
| Td | 露点温度 |
| RRR | 降水量 |
| VV | 能见度 |
| sss | 积雪深度 |

## 项目结构

```
experiment-02-weather-prediction/
├── data/                    # 原始气象数据 (.xls, gitignored)
├── notebooks/               # 数据探索与可视化
├── src/                     # 建模与训练脚本
│   ├── preprocess.py        # 数据清洗、特征工程、时间切分
│   ├── train_xgboost.py     # XGBoost 基线模型
│   └── train_lstm.py        # LSTM 深度学习模型
├── reports/
│   ├── experiment_proposal.md   # 问题规划与方向分析
│   └── experiment_report.md     # (待填写)
├── outputs/                 # 预处理后的 CSV 和结果
└── requirements.txt         # Python 依赖
```

## 环境依赖

```bash
pip install pandas numpy matplotlib scikit-learn
# 或根据建模方案补充：torch, tensorflow, xgboost, prophet 等
```

## 已确定方案

| 决策项 | 选择 | 说明 |
|--------|------|------|
| **预测目标** | 气温 T | 高完整度列（99.99%），周期性强，适合时序建模 |
| **任务类型** | 回归 | 预测未来 3 小时的温度值 |
| **特征** | 多变量 | T, U, Ff, Po, Td + 时间周期性特征 + 滞后特征 |
| **模型方案** | XGBoost + LSTM | 双模型对比：传统 ML vs 深度学习 |
| **评估指标** | RMSE, MAE, R² | 回归标准指标 |
| **验证方式** | 时间序列切分 | Train: 2005-2020, Val: 2021-2022, Test: 2023-2025 |

## 执行流程

```bash
# 1. 数据预处理（构造特征、时间切分）
cd src
python preprocess.py --target T --city all

# 2. XGBoost 基线训练
python train_xgboost.py --target T --n-estimators 500

# 3. LSTM 训练
python train_lstm.py --target T --seq-len 24 --epochs 50

# 4. 对比两个模型的 Test R²，选择最优方案填入报告
```
