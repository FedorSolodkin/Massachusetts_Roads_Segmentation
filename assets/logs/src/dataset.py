import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Tuple, List

class RoadsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        images_dir: str = "images",
        masks_dir: str = "masks",
        transform: Optional[A.Compose] = None,
        img_size: int = 512,
    ):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, images_dir)
        self.masks_dir = os.path.join(root_dir, masks_dir)
        self.img_size = img_size
        self.transform = transform
        
       
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))
        ])
        print(f"📂 Found {len(self.image_files)} images in {root_dir}")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_name = self.image_files[idx]
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask_color = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask_color, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask
def get_train_transform(img_size: int = 512) -> A.Compose:
    """Аугментации для обучения"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        A.GaussNoise(p=0.2, var_limit=(10.0, 50.0)),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet mean
            std=(0.229, 0.224, 0.225),    # ImageNet std
        ),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


def get_val_transform(img_size: int = 512) -> A.Compose:
    """Аугментации для валидации (без случайных преобразований)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 8,
    img_size: int = 512,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Создает DataLoaders для обучения и валидации"""
    
    train_dataset = RoadsDataset(
        root_dir=train_dir,
        transform=get_train_transform(img_size),
        img_size=img_size,
    )
    
    val_dataset = RoadsDataset(
        root_dir=val_dir,
        transform=get_val_transform(img_size),
        img_size=img_size,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Быстрый тест датасета
    print("🧪 Testing RoadsDataset with TIFF files...")
    
    # Проверяем, есть ли данные
    test_dirs = ["data/train", "data/val", "data/test"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"\n📂 Testing {test_dir}...")
            try:
                dataset = RoadsDataset(
                    root_dir=test_dir,
                    img_size=512,
                )
                
                if len(dataset) > 0:
                    img, mask = dataset[0]
                    print(f"   ✅ Image shape: {img.shape}")
                    print(f"   ✅ Mask shape: {mask.shape}")
                    print(f"   ✅ Mask range: [{mask.min():.2f}, {mask.max():.2f}]")
                    print(f"   ✅ Unique mask values: {torch.unique(mask)}")
                else:
                    print(f"   ⚠️ Dataset is empty")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ⚠️ {test_dir} not found")