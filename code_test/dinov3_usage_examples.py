#!/usr/bin/env python3
"""
DINOv3 实际使用示例
展示如何在真实项目中使用DINOv3进行各种任务
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from PIL import Image
from torchvision import transforms
import json
import os
from pathlib import Path

# 添加DINOv3模块路径
sys.path.insert(0, '/home/ipbhk/data/projects/eomt/models/dinov3')

class DINOv3FeatureExtractor:
    """DINOv3特征提取器 - 用于实际项目的封装类"""
    
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = self._setup_transforms()
    
    def _load_model(self):
        """加载模型"""
        weights_path = '/home/ipbhk/data/projects/eomt/models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        repo_dir = '/home/ipbhk/data/projects/eomt/models/dinov3'
        
        model = torch.hub.load(
            repo_dir, 
            'dinov3_vitl16', 
            source='local', 
            weights=weights_path
        )
        return model.to(self.device).eval()
    
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
    
    def extract_features(self, image_path):
        """提取单张图像的特征"""
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(input_tensor)
            
            return features.cpu().numpy()
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return None
    
    def batch_extract_features(self, image_paths, batch_size=8):
        """批量提取特征"""
        features_list = []
        valid_paths = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_valid_paths = []
            
            # 加载批次图像
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(self.transform(image))
                    batch_valid_paths.append(path)
                except Exception as e:
                    print(f"跳过无效图像: {path} ({e})")
            
            if batch_images:
                # 批量处理
                batch_tensor = torch.stack(batch_images).to(self.device)
                with torch.no_grad():
                    batch_features = self.model(batch_tensor)
                
                features_list.extend(batch_features.cpu().numpy())
                valid_paths.extend(batch_valid_paths)
        
        return np.array(features_list), valid_paths

# 使用示例1: 图像相似性搜索
def example_1_image_similarity_search():
    """示例1: 构建图像相似性搜索系统"""
    print("\n=== 示例1: 图像相似性搜索 ===")
    
    extractor = DINOv3FeatureExtractor()
    
    # 假设您有一个图像数据库
    database_images = [
        # 在这里添加您的图像路径
        # "/path/to/your/database/image1.jpg",
        # "/path/to/your/database/image2.jpg",
    ]
    
    if not database_images:
        print("请在代码中添加您的图像路径来测试此功能")
        return
    
    # 1. 为数据库中的所有图像提取特征
    print("正在为数据库图像提取特征...")
    db_features, valid_paths = extractor.batch_extract_features(database_images)
    
    # 2. 保存特征数据库
    feature_db = {
        'features': db_features.tolist(),
        'paths': valid_paths
    }
    
    with open('image_features_db.json', 'w') as f:
        json.dump(feature_db, f)
    
    print(f"特征数据库已保存，包含 {len(valid_paths)} 张图像")
    
    # 3. 搜索相似图像
    def search_similar_images(query_image_path, top_k=5):
        """搜索最相似的图像"""
        query_features = extractor.extract_features(query_image_path)
        if query_features is None:
            return []
        
        # 计算相似性
        similarities = []
        for i, db_feature in enumerate(db_features):
            sim = np.dot(query_features.flatten(), db_feature.flatten()) / (
                np.linalg.norm(query_features) * np.linalg.norm(db_feature)
            )
            similarities.append((sim, valid_paths[i]))
        
        # 排序并返回top-k
        similarities.sort(reverse=True)
        return similarities[:top_k]
    
    print("图像相似性搜索系统构建完成!")
    return search_similar_images

# 使用示例2: 图像分类特征提取
def example_2_classification_features():
    """示例2: 为分类任务提取特征"""
    print("\n=== 示例2: 分类特征提取 ===")
    
    extractor = DINOv3FeatureExtractor()
    
    # 假设您有分类数据集
    dataset_structure = {
        'train': {
            'class1': [],  # 类别1的图像路径列表
            'class2': [],  # 类别2的图像路径列表
            # 更多类别...
        },
        'val': {
            'class1': [],
            'class2': [],
            # 更多类别...
        }
    }
    
    def extract_classification_features(dataset_dict, split='train'):
        """为分类任务提取特征"""
        features_by_class = {}
        
        for class_name, image_paths in dataset_dict[split].items():
            if not image_paths:
                continue
                
            print(f"处理类别: {class_name}")
            class_features, valid_paths = extractor.batch_extract_features(image_paths)
            
            features_by_class[class_name] = {
                'features': class_features,
                'paths': valid_paths
            }
        
        return features_by_class
    
    # 使用示例
    # train_features = extract_classification_features(dataset_structure, 'train')
    # val_features = extract_classification_features(dataset_structure, 'val')
    
    print("分类特征提取函数已准备就绪!")
    print("使用方法:")
    print("1. 在dataset_structure中填入您的数据集路径")
    print("2. 调用extract_classification_features()提取特征")
    print("3. 使用提取的特征训练分类器（如SVM、随机森林等）")

# 使用示例3: 图像聚类
def example_3_image_clustering():
    """示例3: 基于DINOv3特征的图像聚类"""
    print("\n=== 示例3: 图像聚类 ===")
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("请安装scikit-learn: pip install scikit-learn")
        return
    
    extractor = DINOv3FeatureExtractor()
    
    def cluster_images(image_paths, n_clusters=5, use_pca=True, pca_components=128):
        """对图像进行聚类"""
        print(f"正在提取 {len(image_paths)} 张图像的特征...")
        features, valid_paths = extractor.batch_extract_features(image_paths)
        
        if len(features) == 0:
            print("没有成功提取到特征")
            return None
        
        # 可选：使用PCA降维
        if use_pca and features.shape[1] > pca_components:
            print(f"使用PCA降维到 {pca_components} 维...")
            pca = PCA(n_components=pca_components)
            features = pca.fit_transform(features)
        
        # K-means聚类
        print(f"执行K-means聚类，k={n_clusters}...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # 计算聚类质量
        silhouette_avg = silhouette_score(features, cluster_labels)
        print(f"聚类轮廓系数: {silhouette_avg:.4f}")
        
        # 整理结果
        clusters = {}
        for i, (path, label) in enumerate(zip(valid_paths, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(path)
        
        return clusters, silhouette_avg
    
    print("图像聚类函数已准备就绪!")
    print("使用方法:")
    print("1. 准备图像路径列表")
    print("2. 调用cluster_images(image_paths, n_clusters=5)")
    print("3. 结果将按聚类分组返回")
    
    return cluster_images

# 使用示例4: 图像质量评估
def example_4_image_quality_assessment():
    """示例4: 使用特征进行图像质量评估"""
    print("\n=== 示例4: 图像质量评估 ===")
    
    extractor = DINOv3FeatureExtractor()
    
    def assess_image_quality(image_paths, reference_images=None):
        """评估图像质量"""
        features, valid_paths = extractor.batch_extract_features(image_paths)
        
        quality_scores = {}
        
        for i, (feature, path) in enumerate(zip(features, valid_paths)):
            # 方法1: 基于特征的统计特性
            feature_std = np.std(feature)
            feature_range = np.max(feature) - np.min(feature)
            
            # 简单的质量分数（可以根据需要调整）
            quality_score = feature_std * feature_range
            quality_scores[path] = quality_score
        
        # 如果有参考图像，计算与参考的相似性
        if reference_images:
            ref_features, ref_paths = extractor.batch_extract_features(reference_images)
            ref_mean = np.mean(ref_features, axis=0)
            
            for i, (feature, path) in enumerate(zip(features, valid_paths)):
                similarity_to_ref = np.dot(feature.flatten(), ref_mean.flatten()) / (
                    np.linalg.norm(feature) * np.linalg.norm(ref_mean)
                )
                quality_scores[path] = (quality_scores[path], similarity_to_ref)
        
        return quality_scores
    
    print("图像质量评估函数已准备就绪!")
    return assess_image_quality

def main():
    """主函数 - 运行所有示例"""
    print("DINOv3 实际使用示例")
    print("=" * 50)
    
    # 运行所有示例
    search_function = example_1_image_similarity_search()
    example_2_classification_features()
    cluster_function = example_3_image_clustering()
    quality_function = example_4_image_quality_assessment()
    
    print("\n" + "=" * 50)
    print("所有示例已准备完成!")
    print("=" * 50)
    
    print("\n您现在可以:")
    print("1. 修改示例中的图像路径来测试功能")
    print("2. 将这些函数集成到您的项目中")
    print("3. 根据需要调整参数和逻辑")
    
    return {
        'search_function': search_function,
        'cluster_function': cluster_function,
        'quality_function': quality_function
    }

if __name__ == "__main__":
    functions = main()
