#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花卉识别预测脚本 - 符合赛题规范
输出格式: filename, category_id, confidence

使用方法:
    python code/predict.py <测试集文件夹> <输出文件路径>
    
示例:
    python code/predict.py ./test_images ./results/submission.csv
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import class_mapping

# 允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TestDataset(Dataset):
    """测试集数据集"""
    
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # 获取所有图片文件
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        self.image_files = []
        
        img_dir_path = Path(img_dir)
        for ext in img_extensions:
            self.image_files.extend(img_dir_path.glob(f'*{ext}'))
            self.image_files.extend(img_dir_path.glob(f'*{ext.upper()}'))
        
        # 只保留文件名并排序
        self.image_files = sorted([f.name for f in self.image_files])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_files[idx])
        
        try:
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.image_files[idx]
        except Exception as e:
            print(f"\n警告: 无法加载图片 {self.image_files[idx]}: {e}")
            # 返回下一张图片
            next_idx = (idx + 1) % len(self.image_files)
            return self.__getitem__(next_idx)


def create_model_resnet18(num_classes=100):
    """创建ResNet18模型"""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def create_model_resnet50(num_classes=100):
    """创建ResNet50模型"""
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def detect_model_type(checkpoint):
    """检测模型类型"""
    state_dict = checkpoint['model_state_dict']
    
    if 'fc.1.weight' in state_dict:
        fc1_size = state_dict['fc.1.weight'].shape[1]
        if fc1_size == 2048:
            return 'resnet50'
        elif fc1_size == 512:
            return 'resnet18'
    
    return 'resnet18'


def get_test_transforms():
    """获取测试集数据预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def predict(model, dataloader, device):
    """预测函数"""
    model.eval()
    predictions = []
    filenames = []
    confidences = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='预测中')
        for images, names in pbar:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            max_probs, predicted = probs.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            filenames.extend(names)
            confidences.extend(max_probs.cpu().numpy())
    
    return filenames, predictions, confidences


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='花卉识别预测')
    
    # 位置参数（符合赛题要求）
    parser.add_argument('test_img_dir', type=str, help='测试图片目录')
    parser.add_argument('output_path', type=str, help='预测结果输出路径 (CSV文件)')
    
    # 可选参数
    parser.add_argument('--model_path', type=str, default='./model/best_model.pth',
                       help='模型文件路径')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("花卉识别预测")
    print("=" * 80)
    print(f"测试集目录: {args.test_img_dir}")
    print(f"输出文件: {args.output_path}")
    print(f"模型路径: {args.model_path}")
    
    # 检查文件
    if not os.path.exists(args.test_img_dir):
        print(f"错误: 测试集目录不存在: {args.test_img_dir}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载类别映射
    num_classes = class_mapping['num_classes']
    idx_to_category_id = {int(k): int(v) for k, v in class_mapping['idx_to_class_id'].items()}
    
    print(f"类别数: {num_classes}")
    
    # 加载模型
    print("\n加载模型...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 检测模型类型
    model_type = detect_model_type(checkpoint)
    print(f"模型类型: {model_type.upper()}")
    
    # 创建对应的模型
    if model_type == 'resnet50':
        model = create_model_resnet50(num_classes=num_classes)
    else:
        model = create_model_resnet18(num_classes=num_classes)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功")
    if 'val_acc' in checkpoint:
        print(f"  验证准确率: {checkpoint['val_acc']:.2f}%")
    
    # 加载测试集
    print(f"\n加载测试集...")
    test_dataset = TestDataset(
        img_dir=args.test_img_dir,
        transform=get_test_transforms()
    )
    
    if len(test_dataset) == 0:
        print(f"错误: 在目录 {args.test_img_dir} 中未找到图片文件")
        return
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"✓ 测试集大小: {len(test_dataset)}")
    
    # 预测
    print(f"\n开始预测...")
    filenames, predictions, confidences = predict(model, test_loader, device)
    
    # 转换预测结果：从索引(0-99)转换为真实的category_id
    category_ids = [idx_to_category_id[pred] for pred in predictions]
    
    # 格式化confidence为两位小数
    confidences_formatted = [round(float(conf), 4) for conf in confidences]
    
    # 创建DataFrame（符合赛题要求的格式）
    results_df = pd.DataFrame({
        'filename': filenames,
        'category_id': category_ids,
        'confidence': confidences_formatted
    })
    
    # 按文件名排序
    results_df = results_df.sort_values('filename').reset_index(drop=True)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    results_df.to_csv(args.output_path, index=False)
    
    print(f"\n✓ 预测结果已保存到: {args.output_path}")
    print(f"✓ 预测样本数: {len(results_df)}")
    
    # 显示统计信息
    print("\n" + "=" * 80)
    print("预测统计")
    print("=" * 80)
    import numpy as np
    confidences_array = np.array(confidences)
    print(f"平均置信度: {confidences_array.mean():.4f}")
    print(f"最高置信度: {confidences_array.max():.4f}")
    print(f"最低置信度: {confidences_array.min():.4f}")
    print(f"预测类别数: {len(set(category_ids))}")
    
    # 显示前5个预测结果
    print("\n前5个预测结果:")
    print(results_df.head())
    
    print("\n" + "=" * 80)
    print("预测完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
