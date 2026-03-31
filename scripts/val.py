"""
Скрипт для валидации обученной модели и генерации предсказаний.
"""

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
import cv2
import numpy as np

from model import create_unet_model
from dataset import RoadsDataset, get_val_transform
from loss import CombinedLoss
from metrics import calculate_iou, calculate_dice
from utils import load_config



def validate_and_predict(model, dataloader, criterion, device, save_predictions=False, 
                         save_dir=None):
    """
    Валидация модели + опциональное сохранение предсказаний.
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for i, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
            total_dice += calculate_dice(outputs, masks)
            
            # Сохраняем предсказания (если нужно)
            if save_predictions and save_dir:
                preds = torch.sigmoid(outputs) > 0.5
                preds = preds.cpu().numpy().astype(np.uint8) * 255
                
                # Сохраняем каждую предсказанную маску
                for j in range(preds.shape[0]):
                    pred_mask = preds[j, 0]  # (H, W)
                    
                    # Получаем имя файла (нужно передать из датасета)
                    if hasattr(dataloader.dataset, 'image_files'):
                        img_name = dataloader.dataset.image_files[i * dataloader.batch_size + j]
                        pred_path = save_dir / f"pred_{img_name}"
                        cv2.imwrite(str(pred_path), pred_mask)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "iou": f"{calculate_iou(outputs, masks):.4f}"
            })
    
    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches,
        "iou": total_iou / n_batches,
        "dice": total_dice / n_batches,
    }


def main(config_path: str, weights_path: str, save_predictions: bool = False):
    """Основная функция валидации"""
    
  
    config = load_config(config_path)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔌 Device: {device}")
    

    data_root = config['data']['root_dir']
    val_dir = os.path.join(data_root, config['data']['val_dir'])
    
   
    val_dataset = RoadsDataset(
        root_dir=val_dir,
        transform=get_val_transform(config['data']['img_size']),
        img_size=config['data']['img_size'],
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
        encoder_weights=None,  # Не нужны, загрузим веса вручную
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes'],
        activation=None,
    ).to(device)
    

    print(f"📥 Загружаем веса из: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Веса загружены (эпоха {checkpoint['epoch']}, IoU: {checkpoint['iou']:.4f})")
    

    criterion = CombinedLoss()

    print("🚀 Запускаем валидацию...")
    
    save_dir = Path(config['logging']['results_dir']) if save_predictions else None
    if save_predictions and save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Предсказания будут сохранены в: {save_dir}")
    
    metrics = validate_and_predict(
        model, val_loader, criterion, device,
        save_predictions=save_predictions,
        save_dir=save_dir
    )
    
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
    print("=" * 50)
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   IoU:  {metrics['iou']:.4f}")
    print(f"   Dice: {metrics['dice']:.4f}")
    print("=" * 50)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate trained model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pth file)')
    parser.add_argument('--save-preds', action='store_true',
                        help='Save predictions to disk')
    args = parser.parse_args()
    
    main(args.config, args.weights, save_predictions=args.save_preds)