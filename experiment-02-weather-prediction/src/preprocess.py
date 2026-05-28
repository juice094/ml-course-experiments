#!/usr/bin/env python3
"""
Weather Data Preprocessing
气象数据预处理脚本

功能：
1. 读取三个城市的 xls 数据
2. 解析时间列，构造时间特征
3. 处理缺失值
4. 构造标签（预测目标）
5. 划分训练/验证/测试集（时间序列切分，禁止随机打乱）
6. 导出清洗后的 CSV
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 配置
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 可用的高完整度列（缺失 < 1%）
HIGH_QUALITY_COLS = ['T', 'Po', 'U', 'Ff', 'DD', 'WW', 'VV']

# 中度可用列（缺失 5-30%）
MEDIUM_QUALITY_COLS = ['Td', 'N', 'Pa', 'Tx', 'Tn']


def load_city_data(city_name, file_name):
    """读取单个城市的气象数据。"""
    file_path = DATA_DIR / file_name
    print(f"Loading {city_name} from {file_path} ...")

    # header=6: 跳过前 6 行元数据（文件头信息）
    df = pd.read_excel(file_path, header=6)

    # 重命名时间列（各城市列名不同，但都是第一列）
    time_col = df.columns[0]
    df.rename(columns={time_col: 'datetime'}, inplace=True)

    # 解析时间字符串，格式: "31.12.2025 23:00"
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M', errors='coerce')

    # 添加城市标识
    df['city'] = city_name

    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def clean_data(df, missing_threshold=0.95):
    """数据清洗：处理缺失值、构造特征。"""
    # 删除时间解析失败的行
    df = df.dropna(subset=['datetime'])

    # 删除完全缺失的列
    df = df.dropna(axis=1, how='all')

    # 删除缺失率过高的列（对建模无意义）
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    if cols_to_drop:
        print(f"  Dropping columns with >{missing_threshold*100:.0f}% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # 对所有剩余列做前向/后向填充（气象数据相邻时刻通常相近）
    # 先尝试将 object 列转为数值（某些数值列可能被误读为字符串）
    for col in df.columns:
        if col not in ['datetime', 'city'] and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 再次删除因转换而变成全空的列，然后整体填充
    df = df.dropna(axis=1, how='all')
    df = df.ffill().bfill()

    # 构造时间特征
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['weekofyear'] = df['datetime'].dt.isocalendar().week.astype(int)

    # 构造周期性时间特征（用正弦/余弦编码，保留周期性）
    # 例如：23:00 和 00:00 在数值上相差 23，但在周期性上只相差 1
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    return df


def create_lag_features(df, target_col='T', lags=[1, 2, 3, 4, 8, 12, 24]):
    """
    构造滞后特征（历史值作为特征）。

    参数:
        target_col: 目标变量列名
        lags: 滞后步数列表（3小时间隔，lag=8 表示 24小时前）

    例如 lag=1: 用上一时刻的温度预测当前时刻的温度
    """
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)

    # 构造滑动窗口统计特征
    df[f'{target_col}_rolling_mean_8'] = df[target_col].shift(1).rolling(window=8).mean()
    df[f'{target_col}_rolling_std_8'] = df[target_col].shift(1).rolling(window=8).std()

    return df


def split_by_time(df, train_end='2020-12-31', val_end='2022-12-31'):
    """
    按时间切分数据集（禁止随机打乱）。

    划分:
        Train: 2005-01 ~ 2020-12 (~16年, 80%)
        Val:   2021-01 ~ 2022-12 (~2年, 10%)
        Test:  2023-01 ~ 2025-12 (~3年, 10%)
    """
    train_df = df[df['datetime'] <= train_end].copy()
    val_df = df[(df['datetime'] > train_end) & (df['datetime'] <= val_end)].copy()
    test_df = df[df['datetime'] > val_end].copy()

    print(f"  Train: {len(train_df)} rows ({train_df['datetime'].min()} ~ {train_df['datetime'].max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['datetime'].min()} ~ {val_df['datetime'].max()})")
    print(f"  Test:  {len(test_df)} rows ({test_df['datetime'].min()} ~ {test_df['datetime'].max()})")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Preprocess weather data')
    parser.add_argument('--target', type=str, default='T', help='预测目标列 (默认: T=气温)')
    parser.add_argument('--city', type=str, default='all', choices=['all', '敦煌', '金昌', '酒泉'],
                        help='使用哪个城市的数据')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 加载数据
    # -------------------------------------------------------------------------
    cities = {
        '敦煌': '敦煌.xls',
        '金昌': '金昌.xls',
        '酒泉': '酒泉.xls',
    }

    if args.city == 'all':
        dfs = []
        for city_name, file_name in cities.items():
            df = load_city_data(city_name, file_name)
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        df_all = load_city_data(args.city, cities[args.city])

    print(f"\nTotal combined rows: {len(df_all)}")

    # -------------------------------------------------------------------------
    # 清洗
    # -------------------------------------------------------------------------
    print("\nCleaning data ...")
    df_all = clean_data(df_all)

    # -------------------------------------------------------------------------
    # 构造滞后特征
    # -------------------------------------------------------------------------
    print(f"Creating lag features for target '{args.target}' ...")
    # 按城市分别构造滞后特征（避免不同城市的数据混在一起）
    if args.city == 'all':
        processed_dfs = []
        for city in df_all['city'].unique():
            city_df = df_all[df_all['city'] == city].copy().sort_values('datetime')
            city_df = create_lag_features(city_df, target_col=args.target)
            processed_dfs.append(city_df)
        df_all = pd.concat(processed_dfs, ignore_index=True)
    else:
        df_all = create_lag_features(df_all, target_col=args.target)

    # 删除因滞后产生的 NaN 行
    df_all = df_all.dropna()
    print(f"After lag features and dropna: {len(df_all)} rows")

    # -------------------------------------------------------------------------
    # 时间切分
    # -------------------------------------------------------------------------
    print("\nSplitting by time ...")
    train_df, val_df, test_df = split_by_time(df_all)

    # -------------------------------------------------------------------------
    # 保存
    # -------------------------------------------------------------------------
    train_path = OUTPUT_DIR / f'train_{args.target}.csv'
    val_path = OUTPUT_DIR / f'val_{args.target}.csv'
    test_path = OUTPUT_DIR / f'test_{args.target}.csv'

    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    # 保存特征列表（供建模脚本读取）
    feature_cols = [c for c in train_df.columns
                    if c not in ['datetime', args.target, 'city', 'WW', 'DD']]
    # 移除原始字符串列，只保留数值特征
    feature_cols = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(train_df[c])]

    feature_info = {
        'target': args.target,
        'feature_cols': feature_cols,
        'categorical_cols': ['city'] if args.city == 'all' else [],
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
    }

    import json
    with open(OUTPUT_DIR / f'feature_info_{args.target}.json', 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {OUTPUT_DIR / f'feature_info_{args.target}.json'}")
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols[:10]}...")


if __name__ == '__main__':
    main()
