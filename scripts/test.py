#!/usr/bin/env python3
"""
Скрипт для тестирования обученной модели сегментации дорог.
Загружает веса, делает предсказания на тестовых данных, сохраняет визуализации.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn.functional as F
import yaml
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import create_unet_model
from dataset import RoadsDataset, get_val_transform
from metrics import calculate_iou, calculate_dice
from utils import load_config


def test_model(config_path: str, weights_path: str, save_predictions: bool = True):
    """
    Тестирование модели на тестовых данных.
    """
    
    # Загружаем конфиг
    config = load_config(config_path)
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔌 Device: {device}")
    
    # Пути к данным
    data_root = config['data']['root_dir']
    test_dir = Path(data_root) / config['data'].get('test_dir', 'test')
    
    if not test_dir.exists():
        print(f"Папка test не найдена, используем val")
        test_dir = Path(data_root) / config['data'].get('val_dir', 'val')
    
    # Создаём датасет
    print(f"Загружаем тестовые данные из: {test_dir}")
    test_dataset = RoadsDataset(
        root_dir=str(test_dir),
        transform=get_val_transform(config['data']['img_size']),
        img_size=config['data']['img_size'],
    )
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    
    # Создаём модель
    print("Создаём модель...")
    model = create_unet_model(
        encoder_name=config['model']['encoder'],
        encoder_weights=None,
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes'],
        activation=None,
    ).to(device)
    
    # Загружаем веса
    print(f"📥 Загружаем веса из: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Веса загружены (эпоха {checkpoint['epoch']}, IoU: {checkpoint['iou']:.4f})")
    
    model.eval()
    
    # Метрики
    total_iou = 0.0
    total_dice = 0.0
    total_loss = 0.0
    
    # Для визуализации
    predictions = []
    images = []
    masks = []
    
    print("🚀 Запускаем тестирование...")
    pbar = tqdm(test_loader, desc="Testing")
    
    from loss import CombinedLoss
    criterion = CombinedLoss()
    with torch.no_grad():
        for batch_idx, (imgs, msks) in enumerate(pbar):
            imgs = imgs.to(device)
            msks = msks.to(device)
            
            # Предсказание
            outputs = model(imgs)
            
            # Loss
            loss = criterion(outputs, msks)
            total_loss += loss.item()
            
            # Метрики
            iou = calculate_iou(outputs, msks)
            dice = calculate_dice(outputs, msks)
            total_iou += iou
            total_dice += dice
            
            # Сохраняем для визуализации (первые 8 изображений)
            if save_predictions and len(predictions) < 8:
                preds = torch.sigmoid(outputs) > 0.5
                predictions.extend(preds.cpu().numpy())
                images.extend(imgs.cpu().numpy())
                masks.extend(msks.cpu().numpy())
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "iou": f"{iou:.4f}",
                "dice": f"{dice:.4f}"
            })
    
    # Итоговые метрики
    n_batches = len(test_loader)
    avg_iou = total_iou / n_batches
    avg_dice = total_dice / n_batches
    avg_loss = total_loss / n_batches
    
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 50)
    print(f"   Test Loss: {avg_loss:.4f}")
    print(f"   Test IoU:  {avg_iou:.4f}")
    print(f"   Test Dice: {avg_dice:.4f}")
    print("=" * 50)
    
    # Сохраняем метрики
    metrics_path = Path(config['logging']['log_dir']) / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    metrics = {
        "test_loss": float(avg_loss),
        "test_iou": float(avg_iou),
        "test_dice": float(avg_dice),
        "checkpoint_epoch": checkpoint['epoch'],
        "checkpoint_iou": float(checkpoint['iou']),
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📁 Метрики сохранены: {metrics_path}")
    
    # Визуализация
    if save_predictions and len(predictions) > 0:
        print("🎨 Сохраняем визуализацию предсказаний...")
        vis_path = Path(config['logging']['log_dir']) / "test_predictions.png"
        
        fig, axes = plt.subplots(len(predictions), 3, figsize=(15, 5 * len(predictions)))
        
        if len(predictions) == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(len(predictions)):
            # Изображение
            img = np.transpose(images[i], (1, 2, 0))
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Image")
            axes[i, 0].axis('off')
            
            # Маска (ground truth)
            gt_mask = masks[i][0]
            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Предсказание
            pred_mask = predictions[i][0]
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📁 Визуализация сохранена: {vis_path}")
        
        # Сохраняем отдельные предсказания для сабмита
        results_dir = Path(config['logging'].get('results_dir', 'assets/results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Сохраняем предсказания в: {results_dir}")
        for i, pred in enumerate(predictions):
            pred_mask = (pred[0] * 255).astype(np.uint8)
            pred_path = results_dir / f"pred_{i:03d}.png"
            cv2.imwrite(str(pred_path), pred_mask)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test road segmentation model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pth file)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save predictions visualization')
    args = parser.parse_args()
    
    test_model(
        config_path=args.config,
        weights_path=args.weights,
        save_predictions=not args.no_save
    )