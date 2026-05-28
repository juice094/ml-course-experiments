#!/usr/bin/env python3
"""
Weather Temperature Prediction - XGBoost Baseline
气象温度预测 - XGBoost 基线模型

任务：多变量回归预测未来时刻温度 T
特征：历史温度滞后值、湿度、气压、风速、时间周期性特征等
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

DATA_DIR = Path(__file__).parent.parent / "outputs"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def load_data(target='T'):
    """加载预处理后的数据。"""
    train_df = pd.read_csv(DATA_DIR / f'train_{target}.csv')
    val_df = pd.read_csv(DATA_DIR / f'val_{target}.csv')
    test_df = pd.read_csv(DATA_DIR / f'test_{target}.csv')

    with open(DATA_DIR / f'feature_info_{target}.json', 'r', encoding='utf-8') as f:
        feature_info = json.load(f)

    feature_cols = feature_info['feature_cols']
    target_col = feature_info['target']

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def evaluate(y_true, y_pred, dataset_name='Dataset'):
    """计算回归评估指标。"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{dataset_name} Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost for temperature prediction')
    parser.add_argument('--target', type=str, default='T', help='Target column')
    parser.add_argument('--n-estimators', type=int, default=500, help='Number of trees')
    parser.add_argument('--max-depth', type=int, default=6, help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--early-stopping', type=int, default=50, help='Early stopping rounds')
    args = parser.parse_args()

    print("Loading data ...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data(args.target)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")

    # -------------------------------------------------------------------------
    # 训练 XGBoost（使用原生 API，兼容 XGBoost 3.x）
    # -------------------------------------------------------------------------
    print("\nTraining XGBoost ...")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
    }

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dval, 'val')],
        early_stopping_rounds=args.early_stopping,
        evals_result=evals_result,
        verbose_eval=False,
    )

    print(f"Best iteration: {model.best_iteration}")

    # -------------------------------------------------------------------------
    # 评估
    # -------------------------------------------------------------------------
    y_pred_train = model.predict(dtrain)
    y_pred_val = model.predict(dval)
    y_pred_test = model.predict(dtest)

    train_metrics = evaluate(y_train, y_pred_train, 'Train')
    val_metrics = evaluate(y_val, y_pred_val, 'Validation')
    test_metrics = evaluate(y_test, y_pred_test, 'Test')

    # -------------------------------------------------------------------------
    # 特征重要性
    # -------------------------------------------------------------------------
    print("\nTop 10 Feature Importances:")
    importance_dict = model.get_score(importance_type='gain')
    # 补全未出现在 importance_dict 中的特征（增益为 0）
    importance_full = {f: importance_dict.get(f, 0.0) for f in feature_cols}
    importance = pd.DataFrame({
        'feature': list(importance_full.keys()),
        'importance': list(importance_full.values()),
    }).sort_values('importance', ascending=False)

    print(importance.head(10).to_string(index=False))

    # -------------------------------------------------------------------------
    # 保存模型和结果
    # -------------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f'xgboost_{args.target}.json'
    model.save_model(str(model_path))

    results = {
        'model': 'XGBoost',
        'target': args.target,
        'config': vars(args),
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        },
        'feature_importance': importance.head(20).to_dict('records'),
    }

    with open(OUTPUT_DIR / f'results_xgboost_{args.target}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Results saved: {OUTPUT_DIR / f'results_xgboost_{args.target}.json'}")


if __name__ == '__main__':
    main()
