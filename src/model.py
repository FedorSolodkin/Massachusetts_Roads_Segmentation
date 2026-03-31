import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional, Dict   

def create_unet_model(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    activation: Optional[str] = "sigmoid",
) -> nn.Module:
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model

def get_model_info(model: nn.Module, input_size: tuple = (1, 3, 512, 512)) -> Dict:
    """
    Выводит информацию о модели (кол-во параметров).
    """
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    # Считаем параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Тестируем проход
    with torch.no_grad():
        output = model(dummy_input)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "input_shape": input_size,
        "output_shape": output.shape,
    }
    
if __name__ == "__main__":
    model = create_unet_model(
        encoder_name="resnet34",
        encoder_weights="imagenet",
    )
    info = get_model_info(model)
    
    
    print(f"✅ U-Net модель создана успешно!")
    print(f"📊 Всего параметров: {info['total_params']:,}")
    print(f"📊 Обучаемых параметров: {info['trainable_params']:,}")
    print(f"📐 Вход: {info['input_shape']}")
    print(f"📐 Выход: {info['output_shape']}")