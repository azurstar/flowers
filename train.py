#!/usr/bin/env python3
"""
训练脚本 (修复 TTA 验证集设备不匹配错误)
"""
import os, argparse, json, torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast
import torch.nn.functional as F
from model import create_model, unfreeze_head, unfreeze_deep_layers, unfreeze_all_layers, smart_load_state_dict
from utils import create_data_loaders, set_seed, AverageMeter, calculate_accuracy, save_config

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train(); losses = AverageMeter(); top1 = AverageMeter()
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        if torch.isnan(images).any(): continue
        outputs = model(images)
        loss = criterion(outputs, labels)
        if torch.isnan(loss): optimizer.zero_grad(); continue
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        acc1, = calculate_accuracy(outputs, labels, topk=(1,))
        losses.update(loss.item(), images.size(0)); top1.update(acc1.item(), images.size(0))
        if batch_idx % 50 == 0: print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss:{losses.avg:.4f} Acc:{top1.avg:.2f}')
    return losses.avg, top1.avg

def validate_epoch(model, val_loader, criterion, device):
    model.eval(); losses = AverageMeter(); top1 = AverageMeter()
    with torch.no_grad():
        # 修复 FutureWarning
        with torch.amp.autocast('cuda', enabled=True):
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                acc1, = calculate_accuracy(outputs, labels, topk=(1,))
                losses.update(criterion(outputs, labels).item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
    print(f'Val (Std): Acc {top1.avg:.2f}%')
    return losses.avg, top1.avg

def validate_epoch_tta(model, val_loader_tta, criterion, device):
    model.eval(); losses = AverageMeter(); top1 = AverageMeter()
    with torch.no_grad():
        # 修复 FutureWarning
        with torch.amp.autocast('cuda', enabled=True):
            for (stacks, labels, _) in val_loader_tta:
                B, N, C, H, W = stacks.shape
                
                # 1. 数据移至 GPU
                flat = stacks.view(B*N, C, H, W).to(device)
                labels = labels.to(device) # <--- 关键修复：labels 必须移至 GPU
                
                # 2. 推理
                outputs_flat = model(flat) # [B*N, NumClasses]
                
                # 3. 计算 Loss (需要将 labels 重复 N 次以匹配 outputs_flat)
                loss = criterion(outputs_flat, labels.repeat_interleave(N))
                
                # 4. 计算 Accuracy (聚合预测结果)
                outputs = outputs_flat.view(B, N, -1)
                probs = F.softmax(outputs, dim=2).mean(dim=1)
                acc1, = calculate_accuracy(probs, labels, topk=(1,))
                
                losses.update(loss.item(), B)
                top1.update(acc1.item(), B)
                
    print(f'Val (TTA): Loss:{losses.avg:.4f} Acc:{top1.avg:.2f}%')
    return losses.avg, top1.avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='./train_labels.csv')
    parser.add_argument('--train_img_dir', default='./train')
    parser.add_argument('--save_dir', default='./model_dinov2')
    parser.add_argument('--val_csv', default=None)
    parser.add_argument('--val_img_dir', default=None)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--model_type', default='vit_base_patch14_dinov2.lvd142m')
    parser.add_argument('--img_size', type=int, default=336)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--val_tta', action='store_true')
    parser.add_argument('--tta_level', type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed); os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda")
    print(f"Device: {device}, ImgSize: {args.img_size}")

    train_dl, val_dl, c2i, n_cls = create_data_loaders(args.train_csv, args.train_img_dir, args.img_size, args.batch_size, args.val_csv, args.val_img_dir, args.val_split, args.seed, args.val_tta, args.tta_level)
    
    # 传递 img_size
    model = create_model(args.model_type, n_cls, img_size=args.img_size).to(device)
    
    crit = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    best_acc = 0.0
    start_ep = 1
    resume_ckpt = None

    if args.resume and os.path.exists(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        resume_ckpt = torch.load(args.resume, map_location=device)
        
        # 使用 smart_load_state_dict 处理分辨率不匹配
        smart_load_state_dict(model, resume_ckpt['model_state_dict'])
        
        start_ep = resume_ckpt['epoch'] + 1
        best_acc = resume_ckpt.get('best_accuracy', 0.0)
        print(f"Resumed from ep {start_ep-1}, best {best_acc:.2f}%")

    opt = None; sched = None
    for ep in range(start_ep, args.epochs+1):
        # Stage Config
        if ep == 1 or (args.resume and ep == start_ep and ep < 4):
            unfreeze_head(model)
            opt = optim.AdamW(model.head.parameters(), lr=1e-3)
            if args.resume and ep == start_ep and resume_ckpt: 
                try: opt.load_state_dict(resume_ckpt['optimizer_state_dict'])
                except: pass
        elif ep == 4 or (args.resume and ep == start_ep and 4 <= ep < 9):
            unfreeze_deep_layers(model)
            opt = optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4}])
            if args.resume and ep == start_ep and resume_ckpt: 
                try: opt.load_state_dict(resume_ckpt['optimizer_state_dict'])
                except: pass
        elif ep == 9 or (args.resume and ep == start_ep and ep >= 9):
            unfreeze_all_layers(model)
            opt = optim.AdamW([{'params':model.backbone.parameters(),'lr':5e-6},{'params':model.head.parameters(),'lr':5e-5}])
            # 修正 T_max
            sched = CosineAnnealingLR(opt, T_max=max(1, args.epochs-9))
            if args.resume and ep == start_ep and resume_ckpt:
                try: 
                    opt.load_state_dict(resume_ckpt['optimizer_state_dict'])
                    skip = ep - 9
                    if skip > 0: 
                        for _ in range(skip): sched.step()
                except: pass
        
        if opt is None: # Fallback for weird resume states
            unfreeze_all_layers(model)
            opt = optim.AdamW(model.parameters(), lr=1e-4)

        train_epoch(model, train_dl, crit, opt, device, ep)
        
        if args.val_tta: v_l, v_a = validate_epoch_tta(model, val_dl, crit, device)
        else: v_l, v_a = validate_epoch(model, val_dl, crit, device)
        
        if sched: sched.step()
        
        # Save
        ckpt = {'epoch':ep, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':opt.state_dict(), 'best_accuracy':max(v_a, best_acc)}
        torch.save(ckpt, os.path.join(args.save_dir, 'latest_checkpoint.pth'))
        torch.save(ckpt, os.path.join(args.save_dir, f'epoch_{ep}.pth')) # Save every epoch
        
        if v_a > best_acc:
            best_acc = v_a
            torch.save(ckpt, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"New Best: {best_acc:.2f}%")

    # Save Config
    # 确保 class_to_idx 的 key 是 int
    c2i_serializable = {int(k): int(v) for k, v in c2i.items()}
    save_config({'model_type':args.model_type, 'num_classes':n_cls, 'img_size':args.img_size, 'class_to_idx':c2i_serializable, 'best_accuracy':best_acc, 'normalization': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}, os.path.join(args.save_dir, 'config.json'))

if __name__ == '__main__': main()