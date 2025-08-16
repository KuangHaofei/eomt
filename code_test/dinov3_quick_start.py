#!/usr/bin/env python3
"""
DINOv3 ViT-L 快速入门示例
最简单的使用方式
"""

import torch
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO

# 添加DINOv3模块路径
sys.path.insert(0, '/home/ipbhk/data/projects/eomt/models/dinov3')

def load_model():
    """加载DINOv3 ViT-L模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 权重文件路径
    weights_path = '/home/ipbhk/data/projects/eomt/models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    
    try:
        # 使用torch.hub加载
        repo_dir = '/home/ipbhk/data/projects/eomt/models/dinov3'
        model = torch.hub.load(
            repo_dir, 
            'dinov3_vitl16', 
            source='local', 
            weights=weights_path
        )
        model = model.to(device).eval()
        print("✓ 模型加载成功!")
        return model, device
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, device

def setup_transforms():
    """设置图像预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

def load_image_from_url(url):
    """从URL加载图像"""
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        print(f"图像加载失败: {e}")
        return None

def extract_features(model, image, transform, device):
    """提取图像特征"""
    if isinstance(image, str):
        if image.startswith('http'):
            image = load_image_from_url(image)
        else:
            image = Image.open(image).convert('RGB')
    
    if image is None:
        return None
    
    # 预处理
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 提取特征
    with torch.no_grad():
        features = model(input_tensor)
    
    return features

def main():
    """主函数"""
    print("DINOv3 ViT-L 快速入门")
    print("=" * 30)
    
    # 1. 加载模型
    model, device = load_model()
    if model is None:
        return
    
    # 2. 设置预处理
    transform = setup_transforms()
    
    # 3. 测试图像
    test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print(f"测试图像: {test_image_url}")
    
    # 4. 提取特征
    features = extract_features(model, test_image_url, transform, device)
    
    if features is not None:
        print(f"✓ 特征提取成功!")
        print(f"  特征维度: {features.shape}")
        print(f"  特征类型: {features.dtype}")
        print(f"  特征范数: {torch.norm(features).item():.4f}")
        print(f"  特征均值: {features.mean().item():.4f}")
        print(f"  特征标准差: {features.std().item():.4f}")
        
        # 显示特征的前10个值
        print(f"  前10个特征值: {features[0, :10].cpu().numpy()}")
        
        print("\n🎉 DINOv3模型运行成功!")
        print("\n您可以使用这些特征进行:")
        print("  - 图像相似性比较")
        print("  - 图像检索")
        print("  - 图像分类")
        print("  - 下游任务的特征提取")
    else:
        print("✗ 特征提取失败")

if __name__ == "__main__":
    main()
