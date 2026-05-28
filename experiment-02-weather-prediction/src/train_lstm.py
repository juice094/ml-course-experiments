#!/usr/bin/env python3
"""
Weather Temperature Prediction - LSTM Model
气象温度预测 - LSTM 深度学习模型

任务：利用历史序列特征，用 LSTM 预测未来时刻温度 T
"""

import json
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader


class Logger:
    """同时将 stdout 输出到终端和日志文件。"""

    def __init__(self, filepath, terminal):
        self.terminal = terminal
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

DATA_DIR = Path(__file__).parent.parent / "outputs"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


class WeatherDataset(Dataset):
    """PyTorch Dataset for time series data."""

    def __init__(self, df, feature_cols, target_col, seq_len=24):
        self.seq_len = seq_len
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x), torch.tensor(y)


class LSTMModel(nn.Module):
    """LSTM + Dense 回归模型。"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # 取最后一个时间步的隐藏状态
        return self.fc(last_hidden).squeeze(-1)


def load_data(target='T', seq_len=24, batch_size=256):
    """加载数据并构造序列。"""
    train_df = pd.read_csv(DATA_DIR / f'train_{target}.csv')
    val_df = pd.read_csv(DATA_DIR / f'val_{target}.csv')
    test_df = pd.read_csv(DATA_DIR / f'test_{target}.csv')

    with open(DATA_DIR / f'feature_info_{target}.json', 'r', encoding='utf-8') as f:
        feature_info = json.load(f)

    feature_cols = feature_info['feature_cols']
    target_col = feature_info['target']

    train_dataset = WeatherDataset(train_df, feature_cols, target_col, seq_len)
    val_dataset = WeatherDataset(val_df, feature_cols, target_col, seq_len)
    test_dataset = WeatherDataset(test_df, feature_cols, target_col, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(feature_cols)


def evaluate_model(model, dataloader, device):
    """评估模型。"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            preds = model(x)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'preds': all_preds, 'targets': all_targets}


def main():
    parser = argparse.ArgumentParser(description='Train LSTM for temperature prediction')
    parser.add_argument('--target', type=str, default='T', help='Target column')
    parser.add_argument('--seq-len', type=int, default=24, help='Sequence length (lag steps)')
    parser.add_argument('--hidden-size', type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()

    # 启动日志重定向
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / f'training_log_lstm_{args.target}_{datetime.now():%Y%m%d_%H%M%S}.txt'
    sys.stdout = Logger(log_path, sys.stdout)
    print(f"Logging to: {log_path}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("Loading data ...")
    train_loader, val_loader, test_loader, input_size = load_data(
        args.target, args.seq_len, args.batch_size
    )
    print(f"Input features: {input_size}, Sequence length: {args.seq_len}")

    # 初始化模型
    model = LSTMModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # 训练
    best_val_rmse = float('inf')
    patience_counter = 0

    print("\nStarting training ...")
    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        val_rmse = val_metrics['rmse']

        scheduler.step(val_rmse)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {np.mean(train_losses):.4f} | "
              f"Val RMSE: {val_rmse:.4f} | "
              f"Val R²: {val_metrics['r2']:.4f}")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            # 保存最佳模型
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': vars(args),
            }, MODEL_DIR / f'lstm_{args.target}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 测试集评估
    print("\nEvaluating on test set ...")
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'model': 'LSTM',
        'target': args.target,
        'config': vars(args),
        'best_val_rmse': best_val_rmse,
        'test_metrics': {k: v for k, v in test_metrics.items() if k not in ('preds', 'targets')},
    }

    with open(OUTPUT_DIR / f'results_lstm_{args.target}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {OUTPUT_DIR / f'results_lstm_{args.target}.json'}")


if __name__ == '__main__':
    main()
