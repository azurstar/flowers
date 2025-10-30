#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型定义文件
包含模型架构、损失函数等
"""
import torch
import torch.nn as nn
from torchvision import transforms, models


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


def create_model(num_classes=100):
    """创建ResNet50模型（更强大）"""
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    
    # 更复杂的分类头
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