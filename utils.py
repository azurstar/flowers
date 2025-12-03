#!/usr/bin/env python3
"""
工具函数 (336px 优化版)
"""
import os, random
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class FlowerDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, class_to_idx=None):
        self.data = dataframe; self.img_dir = img_dir
        self.transform = transform; self.class_to_idx = class_to_idx
        self.data['label_idx'] = self.data['category_id'].map(self.class_to_idx)
        if self.data['label_idx'].isnull().any():
            self.data.dropna(subset=['label_idx'], inplace=True)
            self.data['label_idx'] = self.data['label_idx'].astype(int)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        attempts = 0
        while attempts < 5:
            try:
                row = self.data.iloc[idx]; img_name = row['filename']
                path = os.path.join(self.img_dir, img_name)
                img = Image.open(path).convert('RGB')
                if img.width < 10 or img.height < 10: raise ValueError("Small image")
                label = self.data.iloc[idx]['label_idx']
                if self.transform: 
                    t_img = self.transform(img)
                    if torch.isnan(t_img).any(): raise ValueError("NaN")
                    return t_img, label, img_name
                return img, label, img_name
            except:
                idx = random.randint(0, len(self)-1); attempts += 1
        return torch.zeros((3, 336, 336)), 0, "error.jpg"

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir; self.transform = transform
        self.image_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        path = os.path.join(self.test_dir, img_name)
        try: img = Image.open(path).convert('RGB')
        except: img = Image.new('RGB', (224,224), (0,0,0))
        if self.transform: img = self.transform(img)
        return img, img_name

def get_tta_transforms(img_size, mean, std, tta_level=5):
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if tta_level == 2:
        return [
            transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size), norm]),
            transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.RandomHorizontalFlip(p=1.0), norm])
        ]
    # 5/10 Crop
    # Resize: img_size * 1.15 (e.g. 336 -> 386)
    resize_size = int(img_size * 1.15)
    resizer = transforms.Resize(resize_size)
    if tta_level == 5:
        cropper = transforms.FiveCrop(img_size)
        return lambda img: torch.stack([norm(c) for c in cropper(resizer(img))])
    if tta_level == 10:
        cropper = transforms.TenCrop(img_size)
        return lambda img: torch.stack([norm(c) for c in cropper(resizer(img))])
    return lambda img: norm(transforms.Resize(img_size)(transforms.CenterCrop(img_size)(img)))

class ValTTADataset(Dataset):
    def __init__(self, df, img_dir, img_size, class_to_idx, tta_level):
        self.data = df; self.img_dir = img_dir; self.class_to_idx = class_to_idx; self.tta_level = tta_level
        self.data['label_idx'] = self.data['category_id'].map(self.class_to_idx)
        if self.data['label_idx'].isnull().any(): self.data.dropna(subset=['label_idx'], inplace=True); self.data['label_idx'] = self.data['label_idx'].astype(int)
        self.tta_func = get_tta_transforms(img_size, IMAGENET_MEAN, IMAGENET_STD, tta_level)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]; name = row['filename']
        try: img = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        except: img = Image.new('RGB', (224,224), (0,0,0))
        label = self.data.iloc[idx]['label_idx']
        if self.tta_level == 2: stacked = torch.stack([t(img) for t in self.tta_func])
        else: stacked = self.tta_func(img)
        return stacked, label, name

class TestTTACollectionDataset(Dataset):
    def __init__(self, test_dir, img_size, tta_level):
        self.test_dir = test_dir; self.image_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        self.tta_level = tta_level
        self.tta_func = get_tta_transforms(img_size, IMAGENET_MEAN, IMAGENET_STD, tta_level)
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        name = self.image_files[idx]
        try: img = Image.open(os.path.join(self.test_dir, name)).convert('RGB')
        except: img = Image.new('RGB', (224,224), (0,0,0))
        if self.tta_level == 2: stacked = torch.stack([t(img) for t in self.tta_func])
        else: stacked = self.tta_func(img)
        return stacked, name

def get_transforms(img_size):
    # 增加 RandomErasing 以提升鲁棒性
    train_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return train_transform, val_transform

def get_test_transform(img_size, mean, std):
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def create_data_loaders(train_csv, train_img_dir, img_size, batch_size, val_csv=None, val_img_dir=None, val_split=0.2, seed=42, val_tta=False, tta_level=10):
    df = pd.read_csv(train_csv)
    cats = sorted(df['category_id'].unique())
    c2i = {c: i for i, c in enumerate(cats)}
    print(f"Data: {len(df)} samples, {len(cats)} classes")
    
    tr_t, val_t = get_transforms(img_size)
    
    if val_csv and val_img_dir:
        val_df = pd.read_csv(val_csv); train_df = df; val_dir = val_img_dir
    else:
        train_df, val_df = train_test_split(df, test_size=val_split, random_state=seed, stratify=df['category_id'])
        val_dir = train_img_dir
        
    ds_train = FlowerDataset(train_df, train_img_dir, tr_t, c2i)
    if val_tta: ds_val = ValTTADataset(val_df, val_dir, img_size, c2i, tta_level)
    else: ds_val = FlowerDataset(val_df, val_dir, val_t, c2i)
    
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val, c2i, len(cats)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=0; self.avg=0; self.sum=0; self.count=0
    def update(self, val, n=1): self.val=val; self.sum+=val*n; self.count+=n; self.avg=self.sum/self.count
def calculate_accuracy(outputs, labels, topk=(1,)):
    maxk = max(topk); batch_size = labels.size(0)
    _, pred = outputs.topk(maxk, 1, True, True); pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def save_config(config, path):
    with open(path, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4)
def plot_training_history(history, save_path): pass