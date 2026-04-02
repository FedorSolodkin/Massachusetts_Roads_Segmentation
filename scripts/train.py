import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import os
import json
import csv
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="cv2")
os.environ['OPENCV_IO_SUPPRESS_TIFF_WARNINGS'] = '1'

from model import create_unet_model
from dataset import RoadsDataset, get_train_transform, get_val_transform
from loss import CombinedLoss
from metrics import calculate_iou, calculate_dice
from utils import set_seed, load_config

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks).item()
            total_dice += calculate_dice(outputs, masks).item()
    
    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches,
        "iou": total_iou / n_batches,
        "dice": total_dice / n_batches,
    }

def main(config_path: str):
    config = load_config(config_path)
    set_seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔌 Device: {device}")
    
    data_root = config['data']['root_dir']
    train_dir = os.path.join(data_root, config['data']['train_dir'])
    val_dir = os.path.join(data_root, config['data']['val_dir'])
    
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
    
    print("🏗️ Создаём модель...")
    model = create_unet_model(
        encoder_name=config['model']['encoder'],
        encoder_weights=config['model']['encoder_weights'],
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes'],
        activation=config['model']['activation'],
    ).to(device)
    
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    
    log_dir = Path(config['logging']['log_dir'])
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_config_path = log_dir / f"config_{timestamp}.yaml"
    with open(save_config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"💾 Конфиг сохранён: {save_config_path}")
    
    print("🚀 Начинаем обучение...")
    best_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_iou"].append(float(val_metrics["iou"]))
        history["val_dice"].append(float(val_metrics["dice"]))
        
        print(f"\n📊 Epoch {epoch}:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_metrics['loss']:.4f}")
        print(f"   Val IoU:    {val_metrics['iou']:.4f}")
        print(f"   Val Dice:   {val_metrics['dice']:.4f}")
        
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iou": float(best_iou),
                "config": config,
            }, checkpoint_path)
            print(f"   ⭐ Новая лучшая модель! IoU: {best_iou:.4f}")
    
    print(f"\n✅ Обучение завершено! Лучший IoU: {best_iou:.4f}")
    print(f"📁 Веса сохранены в: {checkpoint_dir / 'best_model.pth'}")
    
    history_path = log_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📊 История сохранена: {history_path}")
    
    history_csv_path = log_dir / "training_history.csv"
    with open(history_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["val_loss"][i],
                history["val_iou"][i],
                history["val_dice"][i]
            ])
    print(f"📊 CSV сохранён: {history_csv_path}")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history["train_loss"]) + 1)
        
        axes[0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
        axes[0].plot(epochs, history["val_loss"], 'r-', label='Val Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(epochs, history["val_iou"], 'g-', label='Val IoU')
        axes[1].plot(epochs, history["val_dice"], 'm-', label='Val Dice')
        axes[1].set_title('Validation Metrics')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = log_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📈 Графики сохранены: {plot_path}")
        plt.close()
    except ImportError:
        print("⚠️ Matplotlib не установлен, графики не сохранены")
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train road segmentation model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)