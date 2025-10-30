#!/usr/bin/env python3
"""
工具函数模块
包含数据处理、可视化、评估等功能
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FlowerDataset(Dataset):
    """花卉数据集类 - 用于有标签的训练数据"""

    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # 动态创建类别ID到索引的映射
        unique_categories = sorted(self.data['category_id'].unique())
        self.class_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}
        self.idx_to_class = {idx: cat_id for cat_id, idx in self.class_to_idx.items()}
        self.num_classes = len(unique_categories)

        print(f"数据集加载: {len(self.data)} 个样本, {len(unique_categories)} 个类别")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误加载图片 {img_path}: {e}")
            # 返回一个空白图像作为备用
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 获取标签索引
        category_id = row['category_id']
        label = self.class_to_idx[category_id]

        return image, label, row['filename']


class ImprovedFlowerDataset(Dataset):
    """改进的花卉数据集类 - 支持更强的数据增强"""
    
    def __init__(self, csv_file, img_dir, is_training=True, img_size=224):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = get_improved_transforms(is_training, img_size)

        # 动态创建类别ID到索引的映射
        unique_categories = sorted(self.data['category_id'].unique())
        self.class_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}
        self.idx_to_class = {idx: cat_id for cat_id, idx in self.class_to_idx.items()}
        self.num_classes = len(unique_categories)

        print(f"改进数据集加载: {len(self.data)} 个样本, {len(unique_categories)} 个类别")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误加载图片 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 获取标签索引
        category_id = row['category_id']
        label = self.class_to_idx[category_id]

        return image, label


class UnlabeledDataset(Dataset):
    """无标签数据集类 - 用于真实预测场景"""

    def __init__(self, img_dir, transform=None, img_extensions=None):
        if img_extensions is None:
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

        self.img_dir = img_dir
        self.transform = transform

        # 获取所有图片文件
        self.image_files = []
        for ext in img_extensions:
            self.image_files.extend(Path(img_dir).glob(f'*{ext}'))
            self.image_files.extend(Path(img_dir).glob(f'*{ext.upper()}'))

        # 排序确保一致性
        self.image_files = sorted([str(f.name) for f in self.image_files])

        print(f"找到 {len(self.image_files)} 张图片用于预测")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, filename)

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误加载图片 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, filename


def get_transforms(phase='train', img_size=224):
    """获取标准数据变换"""
    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


def get_improved_transforms(is_training=True, img_size=224):
    """获取改进的数据增强（更强的增强）"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(train_csv, test_csv, train_img_dir, test_img_dir,
                       batch_size=32, num_workers=4, img_size=224, use_improved=False):
    """创建数据加载器"""

    # 获取变换
    if use_improved:
        train_transform = get_improved_transforms(True, img_size)
        test_transform = get_improved_transforms(False, img_size)
    else:
        train_transform = get_transforms('train', img_size)
        test_transform = get_transforms('test', img_size)

    # 创建数据集
    train_dataset = FlowerDataset(train_csv, train_img_dir, train_transform)
    test_dataset = FlowerDataset(test_csv, test_img_dir, test_transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset.class_to_idx


def create_improved_data_loaders(train_csv, val_csv, train_img_dir, batch_size=32, 
                                num_workers=4, img_size=224):
    """创建改进的数据加载器（使用更强的数据增强）"""
    
    train_dataset = ImprovedFlowerDataset(train_csv, train_img_dir, True, img_size)
    val_dataset = ImprovedFlowerDataset(val_csv, train_img_dir, False, img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.class_to_idx


def set_seed(seed=42):
    """设置随机种子确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")


class AverageMeter:
    """记录平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets, topk=(1, 5)):
    """计算Top-K准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_device():
    """设置训练设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device


def save_config(config, save_path):
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 深度复制配置并转换所有 numpy 类型为 Python 原生类型
    config_serializable = convert_numpy_types(config)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_serializable, f, indent=4, ensure_ascii=False)
    print(f"配置文件已保存到: {save_path}")


def convert_numpy_types(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, (np.integer, np.int64)) else k: convert_numpy_types(v) 
                for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_training_history(history, filepath):
    """保存训练历史"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"训练历史已保存到: {filepath}")


def load_training_history(filepath):
    """加载训练历史"""
    with open(filepath, 'r', encoding='utf-8') as f:
        history = json.load(f)
    return history


def plot_training_history(history, save_path=None):
    """绘制训练历史曲线"""
    # 设置英文字体
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    
    if isinstance(history, list):
        # 新格式：列表中的字典
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        val_acc = [h['val_acc'] for h in history]
    else:
        # 旧格式：字典中的列表
        epochs = range(1, len(history['train_loss']) + 1)
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        train_acc = history['train_acc']
        val_acc = history['val_acc']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_acc, 'r-', label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def calculate_metrics(model, dataloader, device, class_to_idx):
    """计算模型评估指标"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  # 图像, 标签, 文件名
                images, labels, _ = batch
            else:  # 图像, 标签
                images, labels = batch
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 分类报告
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'classification_report': report,
        'confusion_matrix': cm,
        'accuracy': report['accuracy']
    }


def plot_confusion_matrix(cm, class_to_idx, save_path=None):
    """绘制混淆矩阵"""
    # 设置英文字体
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def save_model_checkpoint(state, filename):
    """保存模型检查点"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"模型检查点已保存到: {filename}")


def load_model_checkpoint(model, optimizer=None, filepath='./model/best_model.pth'):
    """加载模型检查点"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"模型已从 epoch {checkpoint['epoch']} 加载，验证准确率: {checkpoint.get('best_acc', 'N/A')}")
    return checkpoint


def evaluate_model(model, test_loader, device, class_to_idx):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # 生成分类报告
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [str(idx_to_class[i]) for i in range(len(class_to_idx))]

    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        output_dict=True
    )

    return accuracy, report, all_preds, all_labels


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


def visualize_predictions(model, dataset, device, class_to_idx, num_samples=8):
    """可视化预测结果"""
    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    # 随机选择样本
    indices = random.sample(range(len(dataset)), num_samples)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            if len(dataset[idx]) == 3:  # 图像, 标签, 文件名
                image, true_label, filename = dataset[idx]
            else:  # 图像, 标签
                image, true_label = dataset[idx]
                filename = f"sample_{idx}"
                
            image_tensor = image.unsqueeze(0).to(device)

            # 预测
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()

            # 转换为显示用的图像
            image_np = image.permute(1, 2, 0).numpy()
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image_np = np.clip(image_np, 0, 1)

            # 显示
            color = 'green' if true_label == predicted_label else 'red'
            axes[i].imshow(image_np)
            axes[i].set_title(f'True: {idx_to_class[true_label]}\nPred: {idx_to_class[predicted_label]}', 
                            color=color, fontsize=10)
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    set_seed(42)
    
    # 测试设备设置
    device = setup_device()
    print(f"设备: {device}")
    
    print("工具函数测试完成!")