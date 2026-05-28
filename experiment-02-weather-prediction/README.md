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
│   └── (待实现)
├── reports/                 # 实验报告
│   └── experiment_report.md # (待创建)
└── requirements.txt         # Python 依赖
```

## 环境依赖

```bash
pip install pandas numpy matplotlib scikit-learn
# 或根据建模方案补充：torch, tensorflow, xgboost, prophet 等
```

## 待确定事项

- [ ] 预测目标：温度 T / 降水量 RRR / 多变量预测？
- [ ] 模型方案：ARIMA / LSTM / Transformer / XGBoost？
- [ ] 评估指标：RMSE / MAE / MAPE？
- [ ] 预测粒度：单步预测 / 多步预测？
