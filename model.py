#!/usr/bin/env python3
"""
模型定义模块 (增强版: 支持不同分辨率权重的自动插值加载)
- 解决 518px 权重加载到 336px 模型时的 size mismatch 问题
- 包含 smart_load_state_dict 工具函数
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.hub import download_url_to_file
import math

# 预训练权重配置
WEIGHT_URLS = {
    'vit_base_patch14_dinov2.lvd142m': {
        'url': 'https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m/resolve/main/pytorch_model.bin',
        'filename': 'vit_base_patch14_dinov2.lvd142m.pth'
    },
    'tf_efficientnet_b5_ns': {
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth',
        'filename': 'tf_efficientnet_b5_ns-6f26d0cf.pth'
    }
}

def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1):
    """
    对 ViT 的位置编码进行插值调整
    posemb: 原始权重中的位置编码 [1, N_old, C]
    posemb_new: 当前模型的位置编码 [1, N_new, C]
    """
    # 维度一致，直接返回
    if posemb.shape == posemb_new.shape:
        return posemb

    ntok_new = posemb_new.shape[1]
    
    # 分离 CLS token (和 registers 如果有)
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = None, posemb

    gs_old = int(math.sqrt(len(posemb_grid[0])))
    gs_new = int(math.sqrt(ntok_new))
    
    # 检查是否是方形 grid
    if gs_old * gs_old != len(posemb_grid[0]) or gs_new * gs_new != ntok_new:
        # 如果无法简单 Reshape，则无法处理，返回原值让 PyTorch 报错
        print(f"Warning: Grid size check failed. Old: {len(posemb_grid[0])}, New: {ntok_new}")
        return posemb

    # [1, N, C] -> [1, H, W, C] -> [1, C, H, W]
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    
    # 双线性插值
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
    
    # [1, C, H, W] -> [1, H, W, C] -> [1, N, C]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    else:
        posemb = posemb_grid
        
    print(f"[Model] Resized pos_embed from {gs_old}x{gs_old} to {gs_new}x{gs_new}")
    return posemb

def smart_load_state_dict(model, state_dict):
    """
    智能加载权重，处理位置编码尺寸不匹配问题
    """
    model_dict = model.state_dict()
    
    # 1. 处理 pos_embed 冲突
    for k in ['backbone.pos_embed', 'pos_embed']:
        if k in state_dict and k in model_dict:
            if state_dict[k].shape != model_dict[k].shape:
                # print(f"Resizing {k}: {state_dict[k].shape} -> {model_dict[k].shape}")
                state_dict[k] = resize_pos_embed(state_dict[k], model_dict[k])

    # 2. 过滤掉形状不匹配的其他层 (例如分类头变化)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                new_state_dict[k] = v
            else:
                pass 
                # print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_dict[k].shape}")
        else:
            pass # 忽略多余的键
            
    # 3. 加载
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    # if len(missing) > 0: print(f"Missing keys (safe to ignore if fine-tuning): {len(missing)}")

class FlowerClassifier(nn.Module):
    def __init__(self, model_name, num_classes, img_size=224, pretrained=True):
        super().__init__()
        self.model_name = model_name.lower()
        self.img_size = img_size
        
        # 1. 创建骨干
        self.backbone = timm.create_model(
            model_name, 
            pretrained=False, 
            num_classes=0, 
            global_pool='',
            img_size=img_size,
            dynamic_img_size=True 
        )
        
        # 2. 智能加载权重 (应用插值逻辑)
        if pretrained:
            self._load_pretrained_weights(model_name)

        # 3. 获取特征维度
        self.in_features = self._detect_feature_dim()

        # 4. MLP Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def _load_pretrained_weights(self, model_name):
        weight_dir = './weights'
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        target_config = None
        for key in WEIGHT_URLS:
            if key in model_name:
                target_config = WEIGHT_URLS[key]; break
        
        if target_config:
            weight_path = os.path.join(weight_dir, target_config['filename'])
            if not os.path.exists(weight_path):
                try: download_url_to_file(target_config['url'], weight_path, progress=False)
                except: pass

            if os.path.exists(weight_path):
                print(f"[Info] Loading base weights: {weight_path}")
                try:
                    checkpoint = torch.load(weight_path, map_location='cpu')
                    # 过滤 + 智能加载
                    state_dict = {k: v for k, v in checkpoint.items() 
                                  if not k.startswith('head.') and not k.startswith('classifier')}
                    
                    # 使用 smart_load_state_dict 来处理这里潜在的尺寸差异
                    smart_load_state_dict(self.backbone, state_dict)
                    
                except Exception as e:
                    print(f"[Error] Base weight load failed: {e}")

    def _detect_feature_dim(self):
        H, W = self.img_size, self.img_size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W)
            try: feat = self.backbone.forward_features(dummy)
            except: feat = self.backbone(dummy)
            if len(feat.shape) == 3: return feat.shape[2]
            return feat.shape[1]

    def forward(self, x):
        try: x = self.backbone.forward_features(x)
        except: x = self.backbone(x)

        if len(x.shape) == 3: 
            if 'dino' in self.model_name or 'vit' in self.model_name:
                x = x[:, 0, :] 
            else:
                x = x.mean(dim=1)
        elif len(x.shape) == 4:
            x = x.mean(dim=[2, 3])
        
        x = self.head(x)
        return x

def create_model(model_name='vit_base_patch14_dinov2.lvd142m', num_classes=100, img_size=224, pretrained=True):
    return FlowerClassifier(model_name, num_classes, img_size=img_size, pretrained=pretrained)

def load_model(model_path, config, device='cpu'):
    model_name = config.get('model_type', 'vit_base_patch14_dinov2.lvd142m')
    num_classes = config['num_classes']
    img_size = config.get('img_size', 224) 
    
    model = create_model(model_name, num_classes, img_size=img_size, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 使用智能加载
    smart_load_state_dict(model, state_dict)
    
    model.to(device)
    model.eval()
    return model

# --- 辅助函数 ---
def freeze_all(model):
    for param in model.parameters(): param.requires_grad = False
def unfreeze_head(model):
    freeze_all(model)
    for param in model.head.parameters(): param.requires_grad = True
def unfreeze_deep_layers(model):
    freeze_all(model)
    for param in model.head.parameters(): param.requires_grad = True
    for name, param in model.backbone.named_parameters():
        for t in ['blocks.8', 'blocks.9', 'blocks.10', 'blocks.11', 'norm', 'blocks.4', 'blocks.5', 'blocks.6']:
            if t in name: param.requires_grad = True; break
def unfreeze_all_layers(model):
    for param in model.parameters(): param.requires_grad = True