#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建验证集脚本 - 从训练集中随机分割验证集
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_validation_set():
    """从训练集中创建验证集"""
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 读取训练标签
    print("读取训练标签...")
    train_df = pd.read_csv('train_labels.csv')
    
    print(f"原始训练集大小: {len(train_df)}")
    print(f"类别数量: {train_df['category_id'].nunique()}")
    
    # 检查数据分布
    class_distribution = train_df['category_id'].value_counts().sort_index()
    print("\n类别分布:")
    for category_id, count in class_distribution.items():
        print(f"类别 {category_id}: {count} 样本")
    
    # 使用分层抽样来确保每个类别在验证集中都有代表性
    # 分割比例：80% 训练，20% 验证
    train_df_split, val_df = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=42,
        stratify=train_df['category_id']  # 分层抽样
    )
    
    # 保存新的训练集（80%）
    train_df_split.to_csv('../train_labels.csv', index=False)
    print(f"\n新的训练集大小: {len(train_df_split)}")
    
    # 保存验证集（20%）
    val_df.to_csv('../val_labels.csv', index=False)
    print(f"验证集大小: {len(val_df)}")
    
    # 验证类别分布
    print("\n验证集类别分布:")
    val_distribution = val_df['category_id'].value_counts().sort_index()
    for category_id, count in val_distribution.items():
        print(f"类别 {category_id}: {count} 样本")
    
    # 验证分层抽样的效果
    print("\n分层抽样验证:")
    for category_id in sorted(train_df['category_id'].unique()):
        original_count = len(train_df[train_df['category_id'] == category_id])
        train_count = len(train_df_split[train_df_split['category_id'] == category_id])
        val_count = len(val_df[val_df['category_id'] == category_id])
        
        original_pct = original_count / len(train_df) * 100
        train_pct = train_count / len(train_df_split) * 100
        val_pct = val_count / len(val_df) * 100
        
        print(f"类别 {category_id}: 原始{original_pct:.1f}% -> 训练{train_pct:.1f}% -> 验证{val_pct:.1f}%")
    
    print(f"\n验证集已保存到: ../val_labels.csv")
    print(f"新的训练集已保存到: ../train_labels.csv")

def create_small_validation_set(validation_ratio=0.2):
    """创建验证集的替代版本，允许调整验证集比例"""
    
    np.random.seed(42)
    
    print("读取训练标签...")
    train_df = pd.read_csv('./train_labels.csv')
    
    print(f"原始训练集大小: {len(train_df)}")
    
    # 按类别分组
    grouped = train_df.groupby('category_id')
    
    train_samples = []
    val_samples = []
    
    # 对每个类别进行分层抽样
    for category_id, group in grouped:
        n_val = max(1, int(len(group) * validation_ratio))  # 每个类别至少1个验证样本
        indices = np.random.permutation(len(group))
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        # 添加到验证集
        val_samples.extend(group.iloc[val_indices].to_dict('records'))
        # 添加到训练集
        train_samples.extend(group.iloc[train_indices].to_dict('records'))
    
    # 创建DataFrame
    train_df_new = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    
    # 保存文件
    train_df_new.to_csv('../train_labels.csv', index=False)
    val_df.to_csv('../val_labels.csv', index=False)
    
    print(f"\n分割完成:")
    print(f"新的训练集: {len(train_df_new)} 样本")
    print(f"验证集: {len(val_df)} 样本")
    print(f"验证集比例: {len(val_df)/len(train_df)*100:.1f}%")
    
    # 显示每个类别的样本数
    print("\n验证集每个类别的样本数:")
    val_counts = val_df['category_id'].value_counts().sort_index()
    for category_id, count in val_counts.items():
        print(f"类别 {category_id}: {count} 样本")

if __name__ == '__main__':
    print("=" * 60)
    print("创建验证集")
    print("=" * 60)
    
    # 使用方法1：使用sklearn的分层抽样（推荐）
    create_validation_set()
    
    print("\n" + "=" * 60)
    print("验证集创建完成！")
    print("=" * 60)
    
    # 可选：使用方法2
    # print("\n使用替代方法创建验证集...")
    # create_small_validation_set(validation_ratio=0.2)