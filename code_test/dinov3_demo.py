#!/usr/bin/env python3
"""
DINOv3 ViT-L 模型使用演示
支持特征提取、相似性计算和可视化功能
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
import os
import sys

# 添加DINOv3模块路径
sys.path.insert(0, '/home/ipbhk/data/projects/eomt/models/dinov3')

class DINOv3Demo:
    def __init__(self, weights_path=None, device=None):
        """
        初始化DINOv3 ViT-L模型
        
        Args:
            weights_path: 权重文件路径，如果为None则使用默认路径
            device: 设备选择，如果为None则自动选择
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 设置权重路径
        if weights_path is None:
            weights_path = '/home/ipbhk/data/projects/eomt/models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        
        self.weights_path = weights_path
        print(f"加载权重文件: {weights_path}")
        
        # 加载模型
        self.model = self._load_model()
        self.model.eval()
        
        # 设置图像变换
        self.transform = self._setup_transforms()
        
        print("DINOv3 ViT-L 模型初始化完成!")
    
    def _load_model(self):
        """加载DINOv3 ViT-L模型"""
        try:
            # 使用torch.hub加载模型
            repo_dir = '/home/ipbhk/data/projects/eomt/models/dinov3'
            model = torch.hub.load(
                repo_dir, 
                'dinov3_vitl16', 
                source='local', 
                weights=self.weights_path,
                pretrained=True
            )
            model = model.to(self.device)
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("尝试直接从权重文件加载...")
            
            # 备用加载方法
            from models.dinov3.dinov3.hub.backbones import dinov3_vitl16
            model = dinov3_vitl16(pretrained=False)
            
            # 加载权重
            state_dict = torch.load(self.weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device)
            return model
    
    def _setup_transforms(self, img_size=224):
        """设置图像预处理变换"""
        return transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),  # ImageNet标准化参数
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
            print(f"从URL加载图像失败: {e}")
            return None
    
    def load_image_from_path(self, path):
        """从本地路径加载图像"""
        try:
            image = Image.open(path).convert('RGB')
            return image
        except Exception as e:
            print(f"从路径加载图像失败: {e}")
            return None
    
    def extract_features(self, image, return_patch_features=False):
        """
        提取图像特征
        
        Args:
            image: PIL图像或图像路径
            return_patch_features: 是否返回patch级特征
            
        Returns:
            features: 特征向量
        """
        if isinstance(image, str):
            if image.startswith('http'):
                image = self.load_image_from_url(image)
            else:
                image = self.load_image_from_path(image)
        
        if image is None:
            return None
        
        # 预处理图像
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if return_patch_features:
                # 返回所有patch特征
                features = self.model.forward_features(input_tensor)
                return features
            else:
                # 返回全局特征 (CLS token)
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
    
    def visualize_attention_map(self, image, patch_size=16):
        """可视化注意力图"""
        if isinstance(image, str):
            if image.startswith('http'):
                original_image = self.load_image_from_url(image)
            else:
                original_image = self.load_image_from_path(image)
        else:
            original_image = image
        
        if original_image is None:
            return None
        
        # 获取patch特征
        patch_features = self.extract_features(original_image, return_patch_features=True)
        
        if patch_features is None:
            return None
        
        # 处理特征格式 - patch_features可能是字典或张量
        if isinstance(patch_features, dict):
            # 如果是字典，尝试获取主要特征
            if 'x_norm_patchtokens' in patch_features:
                features_tensor = patch_features['x_norm_patchtokens']
            elif 'x_prenorm' in patch_features:
                features_tensor = patch_features['x_prenorm']
            else:
                # 使用第一个可用的特征
                features_tensor = list(patch_features.values())[0]
        else:
            features_tensor = patch_features
        
        # 确保是2D张量 [num_patches, feature_dim]
        if len(features_tensor.shape) == 3:
            features_tensor = features_tensor[0]  # 去掉batch维度
        
        # 如果第一个token是CLS token，跳过它
        if features_tensor.shape[0] == 197:  # 196 patches + 1 CLS for 224x224 image
            features_tensor = features_tensor[1:]  # 去掉CLS token
        
        features_np = features_tensor.cpu().numpy()
        
        # 简单的特征可视化 - 计算特征范数
        feature_norms = np.linalg.norm(features_np, axis=1)
        
        # 重塑为空间维度 (14x14 for 224x224 image with patch_size=16)
        h = w = int(np.sqrt(len(feature_norms)))
        if h * w != len(feature_norms):
            print(f"警告: 无法重塑特征图 {len(feature_norms)} -> {h}x{w}")
            return None
            
        attention_map = feature_norms.reshape(h, w)
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 原图
        axes[0].imshow(original_image)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 注意力图
        im = axes[1].imshow(attention_map, cmap='viridis')
        axes[1].set_title('特征强度图')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        return fig
    
    def demo_basic_usage(self):
        """基本使用演示"""
        print("\n=== DINOv3 ViT-L 基本使用演示 ===")
        
        # 示例图像URL
        sample_urls = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",  # 猫
            "http://images.cocodataset.org/val2017/000000397133.jpg",  # 狗
        ]
        
        print("\n1. 特征提取演示:")
        for i, url in enumerate(sample_urls):
            print(f"处理图像 {i+1}: {url}")
            features = self.extract_features(url)
            if features is not None:
                print(f"  特征维度: {features.shape}")
                print(f"  特征范数: {torch.norm(features).item():.4f}")
            else:
                print("  特征提取失败")
        
        print("\n2. 图像相似性计算:")
        similarity = self.compute_similarity(sample_urls[0], sample_urls[1])
        if similarity is not None:
            print(f"图像相似性: {similarity:.4f}")
        
        print("\n3. 注意力可视化:")
        fig = self.visualize_attention_map(sample_urls[0])
        if fig is not None:
            plt.show()
            print("注意力图已显示")
    
    def batch_process_images(self, image_paths):
        """批量处理图像"""
        features_list = []
        valid_paths = []
        
        print(f"批量处理 {len(image_paths)} 张图像...")
        
        for path in image_paths:
            features = self.extract_features(path)
            if features is not None:
                features_list.append(features)
                valid_paths.append(path)
                print(f"✓ 已处理: {path}")
            else:
                print(f"✗ 处理失败: {path}")
        
        if features_list:
            # 堆叠所有特征
            all_features = torch.cat(features_list, dim=0)
            print(f"批量处理完成，总特征形状: {all_features.shape}")
            
            # 计算特征相似性矩阵
            similarity_matrix = F.cosine_similarity(
                all_features.unsqueeze(1), 
                all_features.unsqueeze(0), 
                dim=2
            )
            
            return all_features, similarity_matrix, valid_paths
        
        return None, None, []

def main():
    """主函数演示"""
    print("DINOv3 ViT-L 模型演示程序")
    print("=" * 50)
    
    # 初始化模型
    demo = DINOv3Demo()
    
    # 运行基本演示
    demo.demo_basic_usage()
    
    # 如果有本地图像，可以进行批量处理演示
    print("\n=== 本地图像批量处理演示 ===")
    print("如果您有本地图像文件，请修改以下路径列表:")
    local_images = [
        # 在这里添加您的本地图像路径
        # "/path/to/your/image1.jpg",
        # "/path/to/your/image2.jpg",
    ]
    
    if local_images and all(os.path.exists(path) for path in local_images):
        features, similarity_matrix, valid_paths = demo.batch_process_images(local_images)
        if features is not None:
            print(f"相似性矩阵形状: {similarity_matrix.shape}")
            print("相似性矩阵:")
            print(similarity_matrix.cpu().numpy())
    else:
        print("未找到有效的本地图像文件，跳过批量处理演示")

if __name__ == "__main__":
    main()
