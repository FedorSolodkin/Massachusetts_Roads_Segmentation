#!/usr/bin/env python3
"""
Скрипт для скачивания и подготовки Massachusetts Roads Dataset.
С отладкой имён файлов.
"""

import os
import shutil
from pathlib import Path
import kagglehub


def download_dataset():
    """Скачивает датасет через KaggleHub"""
    print("📥 Скачиваем Massachusetts Roads Dataset...")
    path = kagglehub.dataset_download("balraj98/massachusetts-roads-dataset")
    print(f"✅ Датасет скачан в: {path}")
    return Path(path)


def prepare_dataset(dataset_path: Path, output_dir: str = "data"):
    """
    Копирует данные с умным сопоставлением имён.
    """
    output_path = Path(output_dir)
    
    tiff_dir = dataset_path / 'tiff'
    if not tiff_dir.exists():
        print(f"❌ Папка tiff/ не найдена")
        return None
    
    splits_config = {
        'train': ('train', 'train_labels'),
        'val': ('val', 'val_labels'),
        'test': ('test', 'test_labels'),
    }
    
    print("📋 Копируем файлы...\n")
    
    for split_name, (images_folder, masks_folder) in splits_config.items():
        src_images = tiff_dir / images_folder
        src_masks = tiff_dir / masks_folder
        
        dst_images = output_path / split_name / 'images'
        dst_masks = output_path / split_name / 'masks'
        
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_masks.mkdir(parents=True, exist_ok=True)
        
        if not src_images.exists() or not src_masks.exists():
            print(f"   ⚠️ {split_name}: папки не найдены\n")
            continue
        
        # Получаем все файлы
        image_files = sorted(list(src_images.glob('*.tif')) + list(src_images.glob('*.tiff')))
        mask_files = sorted(list(src_masks.glob('*.tif')) + list(src_masks.glob('*.tiff')))
        
        print(f"📂 {split_name.upper()}:")
        print(f"   Изображений: {len(image_files)}")
        print(f"   Масок:       {len(mask_files)}")
        
        # Показываем примеры имён
        if image_files and mask_files:
            print(f"   Пример изображения: {image_files[0].name}")
            print(f"   Пример маски:       {mask_files[0].name}")
        
        # Создаём словарь масок
        mask_by_stem = {m.stem: m for m in mask_files}
        mask_by_name = {m.name: m for m in mask_files}
        
        # Пробуем разные стратегии сопоставления
        copied_count = 0
        unmatched_images = []
        
        for img_path in image_files:
            mask_found = False
            mask_path = None
            
            # Стратегия 1: Точное совпадение по имени
            if img_path.name in mask_by_name:
                mask_path = mask_by_name[img_path.name]
                mask_found = True
            
            # Стратегия 2: Совпадение по stem (без расширения)
            elif img_path.stem in mask_by_stem:
                mask_path = mask_by_stem[img_path.stem]
                mask_found = True
            
            # Стратегия 3: Ищем маску, которая начинается с того же префикса
            else:
                for mask_stem, mask_file in mask_by_stem.items():
                    # Проверяем, начинается ли имя маски с имени изображения
                    if mask_stem.startswith(img_path.stem) or img_path.stem.startswith(mask_stem):
                        mask_path = mask_file
                        mask_found = True
                        break
            
            if mask_found and mask_path:
                # Копируем изображение
                dst_img = dst_images / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Копируем маску с тем же именем, что и изображение
                dst_mask = dst_masks / img_path.name
                shutil.copy2(mask_path, dst_mask)
                
                copied_count += 1
            else:
                unmatched_images.append(img_path.name)
        
        print(f"   ✅ Скопировано пар: {copied_count}/{len(image_files)}")
        
        if unmatched_images:
            print(f"   ⚠️ Не найдены маски для {len(unmatched_images)} файлов")
            if len(unmatched_images) <= 5:
                for name in unmatched_images[:3]:
                    print(f"      - {name}")
        
        print()
    
    # Итоговая статистика
    print("=" * 50)
    print("📊 ИТОГОВАЯ СТРУКТУРА:")
    print("=" * 50)
    for split_name in ['train', 'val', 'test']:
        n_images = len(list((output_path / split_name / 'images').glob('*.tif')))
        n_masks = len(list((output_path / split_name / 'masks').glob('*.tif')))
        print(f"   {split_name:5s}: {n_images:3d} изображений, {n_masks:3d} масок")
    
    print(f"\n✅ Датасет готов в: {output_path.absolute()}")
    return output_path


if __name__ == "__main__":
    dataset_path = download_dataset()
    result = prepare_dataset(dataset_path, output_dir="data")
    
    if result:
        print("\n🎉 Всё готово к обучению!")