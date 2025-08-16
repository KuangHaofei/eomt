# DINOv3 ViT-L 模型使用说明

## 概述

本项目包含了使用DINOv3 ViT-L预训练模型的完整示例代码。DINOv3是Meta AI开发的先进的自监督视觉模型，能够提取高质量的图像特征。

## 文件说明

1. **dinov3_quick_start.py** - 快速入门示例，最简单的使用方式
2. **dinov3_demo.py** - 完整功能演示，包含特征提取、相似性计算、可视化等
3. **models/dinov3/** - DINOv3模型源代码
4. **models/dinov3/weights/** - 预训练权重文件

## 环境要求

```bash
# 基本依赖
pip install torch torchvision numpy matplotlib pillow requests

# 如果要使用CUDA加速
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始

### 1. 运行快速入门示例

```bash
cd /home/ipbhk/data/projects/eomt
python dinov3_quick_start.py
```

这个脚本会：
- 加载DINOv3 ViT-L模型
- 从网络下载一张测试图像
- 提取图像特征
- 显示特征信息

### 2. 运行完整功能演示

```bash
python dinov3_demo.py
```

这个脚本包含：
- 特征提取
- 图像相似性计算
- 注意力可视化
- 批量处理功能

## 主要功能

### 1. 特征提取

```python
from dinov3_demo import DINOv3Demo

# 初始化模型
demo = DINOv3Demo()

# 提取特征（支持URL或本地路径）
features = demo.extract_features("path/to/image.jpg")
print(f"特征维度: {features.shape}")  # [1, 1024]
```

### 2. 图像相似性计算

```python
# 计算两张图像的相似性
similarity = demo.compute_similarity("image1.jpg", "image2.jpg")
print(f"相似性得分: {similarity:.4f}")  # 范围: [-1, 1]
```

### 3. 批量处理

```python
# 批量处理多张图像
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
features, similarity_matrix, valid_paths = demo.batch_process_images(image_paths)
```

### 4. 注意力可视化

```python
# 可视化模型关注的区域
fig = demo.visualize_attention_map("image.jpg")
plt.show()
```

## 模型规格

- **模型**: DINOv3 ViT-L/16
- **参数量**: 300M
- **输入尺寸**: 224×224
- **特征维度**: 1024
- **预训练数据**: LVD-1689M (网络图像数据集)

## 应用场景

1. **图像检索**: 使用特征向量进行相似图像搜索
2. **图像分类**: 作为特征提取器用于分类任务
3. **图像聚类**: 基于特征相似性进行图像聚类
4. **内容理解**: 理解图像的语义内容
5. **迁移学习**: 作为预训练模型用于下游任务

## 性能优化建议

### 1. GPU加速
```python
# 确保使用GPU（如果可用）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
demo = DINOv3Demo(device=device)
```

### 2. 批量处理
```python
# 批量处理多张图像更高效
batch_images = [transform(img).unsqueeze(0) for img in images]
batch_tensor = torch.cat(batch_images, dim=0).to(device)
with torch.no_grad():
    batch_features = model(batch_tensor)
```

### 3. 内存管理
```python
# 处理大量图像时，及时清理内存
torch.cuda.empty_cache()  # 清理GPU内存
```

## 常见问题

### Q1: 模型加载失败？
**A**: 检查权重文件路径是否正确，确保文件存在：
```bash
ls -la /home/ipbhk/data/projects/eomt/models/dinov3/weights/
```

### Q2: CUDA内存不足？
**A**: 减小批量大小或使用CPU：
```python
demo = DINOv3Demo(device='cpu')
```

### Q3: 图像加载失败？
**A**: 确保图像格式支持（JPG、PNG等），检查网络连接（如果使用URL）

## 扩展功能

### 1. 自定义图像预处理
```python
custom_transform = transforms.Compose([
    transforms.Resize((384, 384)),  # 更高分辨率
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
demo.transform = custom_transform
```

### 2. 保存和加载特征
```python
# 保存特征
torch.save(features, 'features.pt')

# 加载特征
features = torch.load('features.pt')
```

### 3. 特征降维
```python
from sklearn.decomposition import PCA

# 使用PCA降维
pca = PCA(n_components=128)
features_reduced = pca.fit_transform(features.cpu().numpy())
```

## 参考资料

- [DINOv3 论文](https://arxiv.org/abs/2508.10104)
- [DINOv3 官方代码](https://github.com/facebookresearch/dinov3)
- [PyTorch Hub文档](https://pytorch.org/hub/)

## 许可证

本代码遵循DINOv3许可证。详见LICENSE.md文件。
