#!/usr/bin/env python3
"""
Apple Quality Classification - Training Script
苹果品质分类 - 训练脚本
8-class classification: fresh, diseased, bruised, rotten, insect_damaged,
                        cracked, wrinkled, black_spot
8分类任务：合格、病变、碰伤、腐烂、虫伤、裂果、褶皱、黑斑
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm

# =============================================================================
# 配置区：定义数据、模型、输出的存放路径
# =============================================================================
# Path(__file__) 获取当前脚本文件路径，.parent.parent 向上回退两级
# 例如：src/train.py -> src/ -> 项目根目录/
DATA_DIR = Path(__file__).parent.parent / "data" / "train"    # 训练数据目录
MODEL_DIR = Path(__file__).parent.parent / "models"           # 模型权重保存目录
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"         # 日志和输出目录

# 类别名称定义（与 data/train/ 下的子目录名一一对应）
CLASS_NAMES = [
    "fresh", "diseased", "bruised", "rotten",
    "insect_damaged", "cracked", "wrinkled", "black_spot"
]

NUM_CLASSES = len(CLASS_NAMES)  # 类别数量 = 8


# =============================================================================
# 数据预处理：定义训练集和验证集的数据增强/变换策略
# =============================================================================
def get_transforms(train=True):
    """
    返回数据变换（transform）管道。

    参数:
        train (bool): True 表示训练阶段的变换（含数据增强），
                      False 表示验证/测试阶段的变换（仅做标准化）。

    原理:
        训练阶段需要数据增强来扩充样本多样性、抑制过拟合；
        验证阶段必须保持确定性，不能用随机增强，否则评估结果不可比。
    """
    if train:
        # 训练阶段变换：数据增强 + 标准化
        return transforms.Compose([
            # Resize(256): 先把图片缩放到 256x256，为后面的 RandomCrop 留边
            transforms.Resize((256, 256)),
            # RandomCrop(224): 从 256x256 中随机裁剪出 224x224
            # 原理：同一张图每次裁剪位置不同，相当于生成新样本
            transforms.RandomCrop(224),
            # RandomHorizontalFlip: 以 50% 概率水平翻转图片
            # 苹果左右翻转后语义不变，适合用此增强
            transforms.RandomHorizontalFlip(p=0.5),
            # RandomRotation: 在 ±15 度范围内随机旋转
            transforms.RandomRotation(degrees=15),
            # ColorJitter: 随机调整亮度(±20%)和对比度(±20%)
            # 模拟不同光照条件下的拍摄效果
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # ToTensor: 将 PIL Image (H x W x C, 像素值 0-255)
            # 转为 PyTorch Tensor (C x H x W, 像素值 0.0-1.0)
            transforms.ToTensor(),
            # Normalize: 按 ImageNet 预训练模型的均值和标准差做标准化
            # 这是迁移学习的关键：必须和预训练模型使用相同的归一化参数
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # 验证/测试阶段变换：只做 Resize + 标准化，不做随机增强
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


# =============================================================================
# 模型构建：加载预训练 ResNet18 并替换分类头
# =============================================================================
def get_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    加载 EfficientNet-B0 模型，并替换分类头以适应本任务的 8 分类。

    参数:
        num_classes (int): 输出类别数，默认 8。
        pretrained (bool): 是否使用 ImageNet 预训练权重。

    模型切换理由 (ResNet18 -> EfficientNet-B0):
        1. EfficientNet-B0 参数量更少 (~5.3M vs ~11.7M)，过拟合风险更低
        2. ImageNet Top-1 准确率更高 (77.38% vs 69.76%)
        3. 使用复合缩放 (Compound Scaling) 策略，特征提取更高效
        4. 在小样本场景下，更轻量的模型配合 Dropout 能有效抑制过拟合
    """
    model = models.efficientnet_b0(
        weights='IMAGENET1K_V1' if pretrained else None
    )

    # EfficientNet-B0 的分类头结构：
    #   model.classifier = Sequential(
    #       Dropout(p=0.2, inplace=True),
    #       Linear(in_features=1280, out_features=1000)
    #   )
    # 我们替换整个 classifier，增加 Dropout(0.5) 防止过拟合
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),  # 训练时随机丢弃 50% 神经元，强制模型更鲁棒
        nn.Linear(in_features, num_classes)
    )

    return model


# =============================================================================
# 训练一个 Epoch
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    执行一个训练 epoch（遍历一遍训练集）。

    参数:
        model: 神经网络模型
        dataloader: 训练数据加载器，每次 yield 一个 batch
        criterion: 损失函数（本实验用 CrossEntropyLoss）
        optimizer: 优化器（本实验用 AdamW）
        device: 运行设备（cuda 或 cpu）

    返回:
        avg_loss (float): 该 epoch 的平均损失
        accuracy (float): 该 epoch 的训练准确率（百分比）

    训练流程（单步）:
        1. 前向传播 (forward): 输入图片 -> 模型 -> 得到预测 logits
        2. 计算损失 (loss): 比较预测和真实标签的差异
        3. 反向传播 (backward): 计算损失对各参数的梯度
        4. 参数更新 (step): 优化器根据梯度调整参数，使损失减小
    """
    model.train()  # 切换到训练模式（启用 Dropout、BatchNorm 等）
    running_loss = 0.0  # 累加损失，用于计算平均
    correct = 0         # 预测正确的样本数
    total = 0           # 总样本数

    # tqdm 显示进度条，desc="Training" 设置进度条标题
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        # 将数据从 CPU 内存移动到 GPU/CPU 设备上
        # .to(device) 是 PyTorch 的显存/内存管理关键操作
        images, labels = images.to(device), labels.to(device)

        # 1. 梯度清零：每次迭代前必须清零，否则梯度会累加
        optimizer.zero_grad()

        # 2. 前向传播：模型输出 shape 为 [batch_size, num_classes]
        # 每个元素表示该样本属于每个类别的"得分"（logits）
        outputs = model(images)

        # 3. 计算损失：CrossEntropyLoss 内部同时做了 softmax 和负对数似然
        # 输入是 logits，目标是对应类别的整数索引（0-7）
        loss = criterion(outputs, labels)

        # 4. 反向传播：计算 loss 对模型每个参数的梯度
        # 这些梯度存储在每个参数的 .grad 属性中
        loss.backward()

        # 5. 参数更新：优化器根据梯度和学习率更新权重
        optimizer.step()

        # 累加统计量（用于显示进度和计算 epoch 平均）
        running_loss += loss.item()

        # outputs.max(1): 在第 1 维（类别维度）上取最大值
        # 返回 (最大值, 最大值的索引)，索引即预测的类别
        _, predicted = outputs.max(1)

        total += labels.size(0)  # 当前 batch 的样本数
        correct += predicted.eq(labels).sum().item()  # 预测正确的数量

        # 更新进度条后缀显示实时 loss 和 acc
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    # 返回 epoch 平均损失和准确率
    return running_loss / len(dataloader), 100. * correct / total


# =============================================================================
# 验证一个 Epoch
# =============================================================================
def validate(model, dataloader, criterion, device):
    """
    在验证集上评估模型性能。

    与 train_epoch 的区别:
        1. model.eval(): 切换到评估模式（禁用 Dropout，BatchNorm 使用训练时的统计量）
        2. torch.no_grad(): 禁用梯度计算，节省显存和计算资源
        3. 不做反向传播和参数更新

    验证的目的:
        评估模型在未见过的数据上的泛化能力，防止过拟合。
        验证准确率最高的模型被保存为 best_model.pth。
    """
    model.eval()  # 切换到评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    # torch.no_grad() 上下文管理器：不计算梯度，加速推理，减少显存占用
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


# =============================================================================
# 主函数：完整的训练流程
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 命令行参数解析
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Train apple quality classifier')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数（遍历完整数据集的次数）')
    parser.add_argument('--batch-size', type=int, default=8, help='每批处理的图片数量')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率：控制参数更新步长的大小')
    parser.add_argument('--val-split', type=float, default=0.2, help='验证集占比（0.2 = 20%）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，保证实验可复现')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 设置随机种子：使随机操作（数据增强、划分、权重初始化）结果可复现
    # -------------------------------------------------------------------------
    torch.manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # 设备选择：优先使用 GPU（CUDA），否则回退到 CPU
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 创建输出目录（如果不存在）
    # -------------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 加载数据集
    # -------------------------------------------------------------------------
    # ImageFolder 会自动扫描 data/train/ 下的子目录：
    #   子目录名 = 类别名，目录下的图片 = 该类别的样本
    # 它会自动分配类别索引：按字母顺序排序，fresh=0, black_spot=1, ...
    print(f"Loading dataset from: {DATA_DIR}")
    full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=get_transforms(train=True))

    # 用 ImageFolder 自动发现的类别名替换预定义的 CLASS_NAMES
    # 确保顺序与实际数据一致（虽然通常字母顺序不变）
    global CLASS_NAMES
    CLASS_NAMES = [full_dataset.classes[i] for i in range(len(full_dataset.classes))]
    print(f"Classes: {CLASS_NAMES}")
    print(f"Total samples: {len(full_dataset)}")

    # -------------------------------------------------------------------------
    # 划分训练集和验证集
    # -------------------------------------------------------------------------
    # random_split: 随机打乱后按比例划分
    # generator=torch.Generator().manual_seed(args.seed): 固定随机种子保证可复现
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # 验证集使用不含数据增强的变换（确定性评估）
    # 注意：val_dataset.dataset 是底层的 ImageFolder 数据集对象
    val_dataset.dataset.transform = get_transforms(train=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # -------------------------------------------------------------------------
    # 创建 DataLoader：负责 batch 化、打乱、并行加载
    # -------------------------------------------------------------------------
    # shuffle=True: 每个 epoch 打乱训练数据顺序，防止模型记忆顺序
    # num_workers=2: 用 2 个子进程并行加载数据（Windows 上可能需改为 0）
    # pin_memory=True: 将数据固定在页锁定内存，加速 GPU 数据传输
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------------------------------------------------------
    # 初始化模型
    # -------------------------------------------------------------------------
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)  # 将模型参数移动到 GPU/CPU
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # 打印模型总参数量（ResNet18 约 1100 万参数）

    # -------------------------------------------------------------------------
    # 损失函数、优化器、学习率调度器
    # -------------------------------------------------------------------------
    # CrossEntropyLoss: 分类任务的标准损失函数
    #   公式: -log(softmax(output)[target])
    #   即：鼓励模型给正确类别更高的概率，给错误类别更低的概率
    #
    # 类别权重：样本越少的类别，权重越高
    #   各类样本数: black_spot=20, bruised=20, cracked=16, diseased=14,
    #              fresh=20, insect_damaged=20, rotten=20, wrinkled=20
    #   目的：防止模型"忽视"样本少的类别（diseased 14张, cracked 16张）
    class_weights = torch.tensor([1.0, 1.0, 1.25, 1.4, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AdamW: Adam 优化器的改进版，对权重衰减（L2 正则化）的实现更正确
    #   lr=0.001: 学习率，控制每次参数更新的步长
    #   weight_decay=5e-4: L2 正则化系数（从 1e-4 增大到 5e-4）
    #   增大原因：ResNet18 baseline 出现严重过拟合（Train Acc 99%, Val Acc 73%）
    #   更强的惩罚迫使模型权重更小、更简单，提升泛化能力
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # CosineAnnealingLR: 余弦退火学习率调度
    #   T_max=50: 50 个 epoch 完成一个余弦周期，学习率从 lr 降到接近 0
    #   原理：训练初期用较大学习率快速收敛，后期用小学习率精细调整
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -------------------------------------------------------------------------
    # TensorBoard：可视化训练过程
    # -------------------------------------------------------------------------
    # time.strftime('%Y%m%d_%H%M%S'): 用时间戳命名，避免覆盖之前的日志
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=OUTPUT_DIR / 'runs' / run_timestamp)

    # -------------------------------------------------------------------------
    # 文本日志和 JSON 指标文件（用于课程提交）
    # -------------------------------------------------------------------------
    log_file = OUTPUT_DIR / f'training_log_{run_timestamp}.txt'
    metrics_file = OUTPUT_DIR / f'metrics_{run_timestamp}.json'

    # -------------------------------------------------------------------------
    # 训练主循环
    # -------------------------------------------------------------------------
    best_val_acc = 0.0  # 记录历史最高验证准确率
    history = []        # 存储每个 epoch 的指标，用于后续导出 JSON

    print(f"\n{'='*50}")
    print("Starting training...")
    print(f"{'='*50}\n")

    # 写入日志表头
    with open(log_file, 'w', encoding='utf-8') as lf:
        lf.write(f"Apple Quality Detection - Training Log\n")
        lf.write(f"Run Timestamp: {run_timestamp}\n")
        lf.write(f"Model: EfficientNet-B0 (pretrained) + Dropout(0.5)\n")
        lf.write(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}\n")
        lf.write(f"Train Samples: {len(train_dataset)}, Val Samples: {len(val_dataset)}\n")
        lf.write(f"Classes: {CLASS_NAMES}\n")
        lf.write(f"Device: {device}\n")
        lf.write("=" * 70 + "\n")
        lf.write(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR':>12} | {'Best':>4}\n")
        lf.write("-" * 70 + "\n")

    # 遍历每个 epoch
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)

        # 执行训练 + 验证
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率：按余弦曲线衰减
        scheduler.step()

        # ---------------------------------------------------------------------
        # TensorBoard 记录：实时可视化训练指标
        # 启动 TensorBoard: tensorboard --logdir=../outputs/runs
        # 然后在浏览器打开 http://localhost:6006
        # ---------------------------------------------------------------------
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # 判断是否为当前最优模型
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        # 控制台输出当前 epoch 结果
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        if is_best:
            print(f"Saved best model (val_acc: {val_acc:.2f}%)")

        # 写入文本日志
        with open(log_file, 'a', encoding='utf-8') as lf:
            lr_val = optimizer.param_groups[0]['lr']
            lf.write(f"{epoch+1:6d} | {train_loss:10.4f} | {train_acc:9.2f}% | "
                     f"{val_loss:10.4f} | {val_acc:9.2f}% | {lr_val:12.6f} | "
                     f"{'*' if is_best else '':4s}\n")

        # 追加到 history 列表，用于后续导出 JSON
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'is_best': is_best,
        })

        # 保存最优模型权重
        # torch.save 保存的是一个字典，包含：
        #   - epoch: 保存时的 epoch 编号
        #   - model_state_dict: 模型参数（只保存权重，不保存整个模型结构）
        #   - optimizer_state_dict: 优化器状态（如需断点续训）
        #   - val_acc: 保存时的验证准确率
        #   - class_names: 类别名称列表（推理时需要）
        if is_best:
            model_path = MODEL_DIR / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
            }, model_path)

    # 关闭 TensorBoard writer
    writer.close()

    # -------------------------------------------------------------------------
    # 导出 JSON 格式的完整训练历史（便于程序化分析和绘图）
    # -------------------------------------------------------------------------
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'run_timestamp': run_timestamp,
            'config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'val_split': args.val_split,
                'seed': args.seed,
            },
            'dataset': {
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
                'classes': CLASS_NAMES,
            },
            'best_val_acc': best_val_acc,
            'history': history,
        }, f, ensure_ascii=False, indent=2)

    # 写入日志尾部总结
    with open(log_file, 'a', encoding='utf-8') as lf:
        lf.write("=" * 70 + "\n")
        lf.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        lf.write(f"Log saved to: {log_file}\n")
        lf.write(f"Metrics saved to: {metrics_file}\n")

    # 保存最终模型（不一定是最好的，是最后一个 epoch 的）
    final_path = MODEL_DIR / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
    }, final_path)

    # 训练结束，输出总结
    print(f"\n{'='*50}")
    print(f"Training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"{'='*50}")


# =============================================================================
# 程序入口
# =============================================================================
# __name__ == '__main__' 确保当脚本被直接运行时执行 main()，
# 而被作为模块导入时不执行（避免意外运行训练）
if __name__ == '__main__':
    main()
