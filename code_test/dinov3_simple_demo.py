#!/usr/bin/env python3
"""
DINOv3 ViT-L 简单演示 (无可视化)
专注于核心功能：特征提取和相似性计算
"""

import torch
import torch.nn.functional as F
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO

# 添加DINOv3模块路径
sys.path.insert(0, '/home/ipbhk/data/projects/eomt/models/dinov3')

class SimpleDINOv3Demo:
    def __init__(self, weights_path=None, device=None):
        """初始化DINOv3 ViT-L模型"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        if weights_path is None:
            weights_path = '/home/ipbhk/data/projects/eomt/models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        
        self.weights_path = weights_path
        print(f"加载权重: {weights_path}")
        
        # 加载模型
        self.model = self._load_model()
        self.model.eval()
        
        # 设置图像变换
        self.transform = self._setup_transforms()
        print("DINOv3 ViT-L 模型初始化完成!")
    
    def _load_model(self):
        """加载模型"""
        repo_dir = '/home/ipbhk/data/projects/eomt/models/dinov3'
        model = torch.hub.load(
            repo_dir, 
            'dinov3_vitl16', 
            source='local', 
            weights=self.weights_path
        )
        return model.to(self.device)
    
    def _setup_transforms(self):
        """设置图像预处理"""
        return transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
    
    def load_image_from_url(self, url):
        """从URL加载图像"""
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"图像加载失败: {e}")
            return None
    
    def extract_features(self, image_input):
        """提取图像特征"""
        if isinstance(image_input, str):
            if image_input.startswith('http'):
                image = self.load_image_from_url(image_input)
            else:
                image = Image.open(image_input).convert('RGB')
        else:
            image = image_input
        
        if image is None:
            return None
        
        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(input_tensor)
        
        return features
    
    def compute_similarity(self, image1, image2):
        """计算两张图像的相似性"""
        features1 = self.extract_features(image1)
        features2 = self.extract_features(image2)
        
        if features1 is None or features2 is None:
            return None
        
        # 计算余弦相似性
        similarity = F.cosine_similarity(features1, features2, dim=1)
        return similarity.item()
    
    def run_demo(self):
        """运行演示"""
        print("\n" + "="*50)
        print("DINOv3 ViT-L 功能演示")
        print("="*50)
        
        # 测试图像
        test_images = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",  # 猫
            "http://images.cocodataset.org/val2017/000000397133.jpg",  # 狗
            "http://images.cocodataset.org/val2017/000000037777.jpg",  # 交通标志
        ]
        
        print("\n1. 特征提取测试:")
        features_list = []
        
        for i, url in enumerate(test_images):
            print(f"\n处理图像 {i+1}: {url}")
            features = self.extract_features(url)
            
            if features is not None:
                features_list.append(features)
                print(f"  ✓ 特征维度: {features.shape}")
                print(f"  ✓ 特征范数: {torch.norm(features).item():.4f}")
                print(f"  ✓ 特征均值: {features.mean().item():.4f}")
                print(f"  ✓ 特征标准差: {features.std().item():.4f}")
            else:
                print("  ✗ 特征提取失败")
        
        print(f"\n成功提取了 {len(features_list)} 张图像的特征")
        
        if len(features_list) >= 2:
            print("\n2. 图像相似性计算:")
            
            # 计算所有图像对的相似性
            for i in range(len(features_list)):
                for j in range(i+1, len(features_list)):
                    similarity = F.cosine_similarity(features_list[i], features_list[j], dim=1)
                    print(f"  图像{i+1} vs 图像{j+1}: {similarity.item():.4f}")
            
            print("\n3. 特征统计信息:")
            all_features = torch.cat(features_list, dim=0)
            print(f"  所有特征形状: {all_features.shape}")
            print(f"  特征范围: [{all_features.min().item():.4f}, {all_features.max().item():.4f}]")
            print(f"  特征均值: {all_features.mean().item():.4f}")
            print(f"  特征标准差: {all_features.std().item():.4f}")
        
        print("\n" + "="*50)
        print("演示完成! 🎉")
        print("="*50)
        
        return features_list

def main():
    """主函数"""
    demo = SimpleDINOv3Demo()
    features = demo.run_demo()
    
    print("\n使用建议:")
    print("1. 您可以使用 demo.extract_features(image_path) 提取任意图像的特征")
    print("2. 使用 demo.compute_similarity(img1, img2) 计算图像相似性")
    print("3. 特征可用于图像检索、分类、聚类等任务")
    print("4. 特征维度为1024，可以直接用于下游机器学习任务")
    
    return demo

if __name__ == "__main__":
    demo = main()
