#!/usr/bin/env python3
"""
模型压缩脚本 (修复验证环节的尺寸匹配问题)
功能:
1. 移除优化器状态 (Optimizer State)
2. 将权重转换为 FP16 (Half Precision)
3. 验证压缩后的模型 (自动读取 img_size 并使用智能加载)
"""

import os
import torch
import argparse
import sys
from pathlib import Path

# 尝试导入 create_model 和 smart_load_state_dict
try:
    from model import create_model, smart_load_state_dict
except ImportError:
    print("Warning: Could not import from model.py. Skipping verification step.")
    create_model = None

def compress(input_path, output_path):
    print(f"正在处理: {input_path}")
    print(f"原始大小: {os.path.getsize(input_path) / 1024 / 1024:.2f} MB")

    # 1. 加载原始 Checkpoint
    # map_location='cpu' 防止爆显存
    checkpoint = torch.load(input_path, map_location='cpu')

    new_checkpoint = {}

    # 2. 提取并转换 model_state_dict
    if 'model_state_dict' in checkpoint:
        print("发现 model_state_dict，正在转换为 FP16...")
        state_dict = checkpoint['model_state_dict']
    else:
        # 假设整个 checkpoint 就是 state_dict
        print("未发现 model_state_dict 键，假设文件仅包含权重，正在转换为 FP16...")
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        # 将 Tensor 转换为 FP16
        if isinstance(v, torch.Tensor):
            new_state_dict[k] = v.half()
        else:
            new_state_dict[k] = v
    
    new_checkpoint['model_state_dict'] = new_state_dict

    # 3. 保留必要的元数据 (如果有)
    # 这一点很重要，我们需要保留 config 以便知道 img_size
    keys_to_keep = ['epoch', 'best_accuracy', 'config', 'class_to_idx']
    for key in keys_to_keep:
        if key in checkpoint:
            new_checkpoint[key] = checkpoint[key]

    # 4. 保存压缩后的模型
    print(f"正在保存至: {output_path}")
    torch.save(new_checkpoint, output_path)
    
    final_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"压缩后大小: {final_size:.2f} MB")
    print(f"体积减少: {(1 - os.path.getsize(output_path)/os.path.getsize(input_path))*100:.1f}%")

    return new_checkpoint

def verify(output_path, checkpoint):
    """验证模型是否能被正常加载"""
    if create_model is None:
        return

    print("\n--- 正在验证压缩模型 ---")
    try:
        # 1. 尝试从 checkpoint 中获取配置信息
        config = checkpoint.get('config', {})
        
        # 获取 img_size (如果找不到，默认为 224，但这可能导致之前的报错)
        # 我们优先信任 config 中的值
        img_size = config.get('img_size', 224)
        num_classes = config.get('num_classes', 152) # 默认152
        model_type = config.get('model_type', 'vit_base_patch14_dinov2.lvd142m')

        print(f"验证参数: model={model_type}, img_size={img_size}, classes={num_classes}")

        # 2. 创建模型 (传入正确的 img_size)
        model = create_model(model_type, num_classes=num_classes, img_size=img_size, pretrained=False)
        
        # 3. 加载 FP16 权重
        # 使用 smart_load_state_dict 来处理任何潜在的尺寸问题 (如 pos_embed)
        smart_load_state_dict(model, checkpoint['model_state_dict'])
        
        print("验证成功！模型可以被正常加载。")
        print("注意: 推理时请确保使用 autocast 配合 FP16 权重。")
        
    except Exception as e:
        print(f"验证失败: {e}")
        import traceback
        traceback.print_exc()
        print("请检查 model.py 是否与权重结构匹配。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='压缩 PyTorch 模型 (Strip Optimizer + FP16)')
    parser.add_argument('input_model', type=str, help='原始 .pth 文件路径')
    parser.add_argument('output_model', type=str, help='输出 .pth 文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_model):
        print(f"错误: 找不到文件 {args.input_model}")
        sys.exit(1)
        
    # 确保输出目录存在
    out_dir = os.path.dirname(args.output_model)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    compressed_ckpt = compress(args.input_model, args.output_model)
    verify(args.output_model, compressed_ckpt)