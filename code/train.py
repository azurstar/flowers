#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版训练脚本 - 提升准确率到85%+
主要改进：
1. 更长的训练时间（50 epochs）
2. 余弦退火学习率调度
3. 标签平滑
4. 更强的数据增强
5. 使用ResNet50替代ResNet18
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入模块
from model import create_model, LabelSmoothingCrossEntropy
from utils import (
    setup_device, save_training_history, save_model_checkpoint, set_seed,
    create_improved_data_loaders, save_config, plot_training_history, 
    calculate_metrics, evaluate_model, plot_confusion_matrix
)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        if len(batch) == 3:  # 图像, 标签, 文件名
            images, labels, _ = batch
        else:  # 图像, 标签
            images, labels = batch
            
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            if len(batch) == 3:  # 图像, 标签, 文件名
                images, labels, _ = batch
            else:  # 图像, 标签
                images, labels = batch
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(dataloader), 100. * correct / total


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='花卉分类模型训练')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='数据集根目录')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        choices=['resnet50'],
                        help='模型类型')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减')
    parser.add_argument('--img_size', type=int, default=224,
                        help='图像尺寸')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./model',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的模型路径')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 设置设备
    device = setup_device()
    
    print("=" * 80)
    print("改进版训练配置")
    print("=" * 80)
    print(f"数据目录: {args.data_dir}")
    print(f"模型类型: {args.model_type}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"图像尺寸: {args.img_size}")
    print(f"随机种子: {args.seed}")
    print(f"保存目录: {args.save_dir}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 数据路径
    train_csv = os.path.join(args.data_dir, 'train_labels.csv')
    val_csv = os.path.join(args.data_dir, 'val_labels.csv')
    train_img_dir = os.path.join(args.data_dir, 'train')
    
    # 加载数据
    print("\n" + "=" * 80)
    print("加载数据集")
    print("=" * 80)
    
    train_loader, val_loader, class_to_idx = create_improved_data_loaders(
        train_csv=train_csv,
        val_csv=val_csv,
        train_img_dir=train_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    num_classes = len(class_to_idx)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"类别数量: {num_classes}")
    
    # 创建改进的配置
    config = {
        'model_type': args.model_type,
        'num_classes': num_classes,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'label_smoothing': 0.1,
        'img_size': args.img_size,
        'class_to_idx': class_to_idx,
    }
    
    # 创建改进的模型
    print("\n" + "=" * 80)
    print("创建模型（ResNet50）")
    print("=" * 80)
    model = create_model(num_classes=config['num_classes'])
    model = model.to(device)
    print("✓ 模型创建完成")
    
    # 标签平滑损失函数
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 每10个epoch重启一次
        T_mult=2,
        eta_min=1e-6
    )
    
    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    train_history = []
    
    if args.resume and os.path.isfile(args.resume):
        print(f"加载检查点 '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_history = checkpoint.get('history', train_history)
        print(f"加载检查点 (epoch {start_epoch}, best_acc: {best_acc:.2f}%)")
    
    # 训练循环
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    
    patience = 100  # 早停耐心值
    patience_counter = 0
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 80)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        # 记录历史
        history = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        }
        train_history.append(history)
        
        # 打印结果
        print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"时间: {epoch_time:.1f}s")
        
        # 保存最新检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'history': train_history,
            'config': config
        }
        save_model_checkpoint(checkpoint, os.path.join(args.save_dir, 'latest_checkpoint.pth'))
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            checkpoint['best_acc'] = best_acc
            save_model_checkpoint(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"\n早停: 验证准确率在{patience}个epoch内没有提升")
            break
    
    # 保存训练历史
    save_training_history(train_history, os.path.join(args.save_dir, 'training_history.json'))
    
    # 保存最终配置文件
    final_config = {
        'model_type': config['model_type'],
        'num_classes': config['num_classes'],
        'img_size': config['img_size'],
        'class_to_idx': {str(k): int(v) for k, v in config['class_to_idx'].items()},  # 转换为字符串键和整数值
        'best_accuracy': best_acc,
        'training_params': {
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'label_smoothing': config['label_smoothing'],
            'total_epochs': len(train_history)
        }
    }
    
    save_config(final_config, os.path.join(args.save_dir, 'config.json'))
    
    # 绘制训练历史
    plot_training_history(train_history, os.path.join(args.save_dir, 'training_history.png'))
    
    # 最终评估
    print("\n" + "=" * 80)
    print("最终评估")
    print("=" * 80)
    
    # 加载最佳模型进行最终评估
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型进行最终评估 (epoch {checkpoint['epoch']})")
    
    # 计算验证集指标
    metrics = calculate_metrics(model, val_loader, device, class_to_idx)
    print(f"最终验证准确率: {metrics['accuracy']:.4f}")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(metrics['confusion_matrix'], class_to_idx, 
                         os.path.join(args.save_dir, 'confusion_matrix.png'))
    
    # 保存测试报告
    with open(os.path.join(args.save_dir, 'test_report.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics['classification_report'], f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"模型已保存到: {args.save_dir}/")
    print(f"- best_model.pth (最佳模型)")
    print(f"- latest_checkpoint.pth (最新检查点)")
    print(f"- config.json (配置文件)")
    print(f"- training_history.json (训练历史)")
    print(f"- training_history.png (训练曲线图)")
    print(f"- test_report.json (测试报告)")
    print(f"- confusion_matrix.png (混淆矩阵)")


if __name__ == '__main__':
    main()