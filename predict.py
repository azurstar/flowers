#!/usr/bin/env python3
"""
预测脚本 (极速版)
- 默认使用 336px
- 默认使用 TTA-5
- 显存优化
"""
import os, argparse, json, pandas as pd, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn.functional as F
from model import load_model
from utils import TestDataset, TestTTACollectionDataset, get_test_transform, IMAGENET_MEAN, IMAGENET_STD
from pathlib import Path


def load_config(path):
    if not os.path.exists(path): raise FileNotFoundError(f"{path} missing")
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def predict_standard(model, loader, device):
    model.eval(); results = []; use_fp16 = torch.cuda.is_available()
    print(f"Standard Prediction (FP16={use_fp16})...")
    with torch.no_grad():
        with autocast(enabled=use_fp16):
            for images, filenames in tqdm(loader):
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, 1)
                for i in range(len(filenames)): results.append((filenames[i], preds[i].item(), conf[i].item()))
    return results

def predict_tta(model, loader, device, tta_level):
    model.eval(); results = []; use_fp16 = torch.cuda.is_available()
    print(f"TTA-{tta_level} Prediction (FP16={use_fp16})...")
    with torch.no_grad():
        with autocast(enabled=use_fp16):
            for image_stacks, filenames in tqdm(loader):
                B, N, C, H, W = image_stacks.shape
                flat = image_stacks.view(B*N, C, H, W).to(device)
                outputs = model(flat).view(B, N, -1)
                probs = F.softmax(outputs, dim=2).mean(dim=1)
                conf, preds = torch.max(probs, 1)
                for i in range(B): results.append((filenames[i], preds[i].item(), conf[i].item()))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str)
    parser.add_argument('output_csv', type=str)
    # 默认 Batch 32，TTA-5
    parser.add_argument('--model_dir', type=str, default=f'{Path(__file__).resolve().parent}/../model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--tta_level', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = load_config(os.path.join(args.model_dir, 'config.json'))
    model = load_model(os.path.join(args.model_dir, 'best_model.pth'), config, device)
    
    # 优先使用 Config 中的尺寸，如果没有则默认 336 (极速)
    img_size = config.get('img_size', 336)
    print(f"Inference Image Size: {img_size}")

    if args.use_tta:
        ds = TestTTACollectionDataset(args.test_dir, img_size, args.tta_level)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        raw = predict_tta(model, loader, device, args.tta_level)
    else:
        tr = get_test_transform(img_size, IMAGENET_MEAN, IMAGENET_STD)
        ds = TestDataset(args.test_dir, transform=tr)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        raw = predict_standard(model, loader, device)

    idx_to_class = {int(v): int(k) for k, v in config['class_to_idx'].items()}
    final = [{'filename': f, 'category_id': idx_to_class.get(p, -1), 'confidence': c} for f, p, c in raw]
    pd.DataFrame(final).sort_values('filename').to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")

if __name__ == '__main__':
    main()