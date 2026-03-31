import sys
from pathlib import Path

# Добавляем src в path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import os
from datetime import datetime

# Наши модули
from model import create_unet_model
from dataset import RoadsDataset, get_train_transform, get_val_transform
from loss import CombinedLoss
from metrics import calculate_iou, calculate_dice
from utils import set_seed, load_config

def train_one_epoch(model, dataloader, omtimizer, criterion, device, epoch):
    model.train()
    total_loss =0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images,mask in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs)
        loss.backward()
        optimizer.step()
        
        total_loss+=loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(dataloader)

def validate(model,dataloader,criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for images,masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(image)
            loss = criterion(outputs,masks)
            
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
            total_dice += calculate_dice(outputs, masks)
    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches,
        "iou": total_iou / n_batches,
        "dice": total_dice / n_batches,
    }    

def main(config_path: str):
    """Основная функция обучения"""
    
    # Загружаем конфиг
    config = load_config(config_path)
    set_seed(config['seed'])
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔌 Device: {device}")
    
    # Пути к данным (адаптируются под Colab)
    data_root = config['data']['root_dir']
    train_dir = os.path.join(data_root, config['data']['train_dir'])
    val_dir = os.path.join(data_root, config['data']['val_dir'])
    
    # Создаём датасеты
    print("📂 Загружаем датасеты...")
    train_dataset = RoadsDataset(
        root_dir=train_dir,
        transform=get_train_transform(config['data']['img_size']),
        img_size=config['data']['img_size'],
    )
    
    val_dataset = RoadsDataset(
        root_dir=val_dir,
        transform=get_val_transform(config['data']['img_size']),
        img_size=config['data']['img_size'],
    )
    
    # Создаём DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    
    # Создаём модель
    print("🏗️ Создаём модель...")
    model = create_unet_model(
        encoder_name=config['model']['encoder'],
        encoder_weights=config['model']['encoder_weights'],
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes'],
        activation=config['model']['activation'],
    ).to(device)
    
    # Loss, Optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Создаём папки для сохранения
    log_dir = Path(config['logging']['log_dir'])
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем конфиг
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_config_path = log_dir / f"config_{timestamp}.yaml"
    with open(save_config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"💾 Конфиг сохранён: {save_config_path}")
    
    # Цикл обучения
    print("🚀 Начинаем обучение...")
    best_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Обучение
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Валидация
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Логирование
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])
        
        print(f"\n📊 Epoch {epoch}:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_metrics['loss']:.4f}")
        print(f"   Val IoU:    {val_metrics['iou']:.4f}")
        print(f"   Val Dice:   {val_metrics['dice']:.4f}")
        
        # Scheduler
        scheduler.step(val_metrics["loss"])
        
        # Сохраняем лучшую модель
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iou": best_iou,
                "config": config,
            }, checkpoint_path)
            print(f"   ⭐ Новая лучшая модель! IoU: {best_iou:.4f}")
    
    print(f"\n✅ Обучение завершено! Лучший IoU: {best_iou:.4f}")
    print(f"📁 Веса сохранены в: {checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train road segmentation model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)