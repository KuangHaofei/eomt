#!/usr/bin/env python3
"""
对比EoMT实现的DINOv3和原版DINOv3的特征输出差异
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'models/dinov3')

from models.vit import ViT
from models.eomt import EoMT

def load_original_dinov3():
    """加载原版DINOv3模型"""
    repo_dir = 'models/dinov3'
    weights_path = 'models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    
    model = torch.hub.load(
        repo_dir, 
        'dinov3_vitl16', 
        source='local', 
        weights=weights_path
    )
    return model

def create_eomt_model():
    """创建EoMT模型"""
    encoder = ViT(img_size=(224, 224))
    model = EoMT(
        encoder=encoder,
        num_classes=80,
        num_q=100,
        num_blocks=4,
        masked_attn_enabled=False  # 关闭masked attention，纯特征提取
    )
    return model

def extract_dinov3_features(model, x):
    """从原版DINOv3提取特征"""
    with torch.no_grad():
        # 使用forward_features获取详细输出
        features = model.forward_features(x)
        
        return {
            'cls_token': features['x_norm_clstoken'],           # [B, 1024]
            'storage_tokens': features['x_storage_tokens'],     # [B, 4, 1024]
            'patch_tokens': features['x_norm_patchtokens'],     # [B, 196, 1024]
            'prenorm': features['x_prenorm'],                   # [B, 201, 1024]
        }

def extract_eomt_features(model, x):
    """从EoMT模型提取特征（在添加query tokens之前）"""
    with torch.no_grad():
        # 手动执行EoMT的前向传播，但在添加query tokens之前停止
        # 预处理
        x_norm = (x - model.encoder.pixel_mean) / model.encoder.pixel_std
        
        # 使用DINOv3的标准token准备方法
        tokens, (H, W) = model.encoder.backbone.prepare_tokens_with_masks(x_norm)
        
        # 运行所有blocks（无masked attention）
        for i, block in enumerate(model.encoder.backbone.blocks):
            # 每个block都重新计算RoPE位置编码
            if model.encoder.backbone.rope_embed is not None:
                rope_or_rope_list = model.encoder.backbone.rope_embed(H=H, W=W)
            else:
                rope_or_rope_list = None
            
            # 直接使用DINOv3的block forward（无mask）
            tokens = block(tokens, rope_or_rope_list)
        
        # 应用最终的norm
        if hasattr(model.encoder.backbone, 'norm'):
            tokens_norm = model.encoder.backbone.norm(tokens)
        else:
            tokens_norm = tokens
        
        # 分离不同类型的tokens
        cls_token = tokens_norm[:, 0]                    # [B, 1024]
        storage_tokens = tokens_norm[:, 1:5]             # [B, 4, 1024]
        patch_tokens = tokens_norm[:, 5:]                # [B, 196, 1024]
        
        return {
            'cls_token': cls_token,
            'storage_tokens': storage_tokens,
            'patch_tokens': patch_tokens,
            'prenorm': tokens,  # 未归一化的完整tokens
        }

def compute_feature_differences(features1, features2, names):
    """计算特征差异"""
    results = {}
    
    for key in features1.keys():
        if key in features2:
            f1 = features1[key]
            f2 = features2[key]
            
            # 确保形状匹配
            if f1.shape != f2.shape:
                print(f"警告: {key} 形状不匹配 - {names[0]}: {f1.shape}, {names[1]}: {f2.shape}")
                continue
            
            # 计算各种差异指标
            mse = F.mse_loss(f1, f2).item()
            mae = F.l1_loss(f1, f2).item()
            
            # 余弦相似性
            f1_flat = f1.flatten()
            f2_flat = f2.flatten()
            cosine_sim = F.cosine_similarity(f1_flat.unsqueeze(0), f2_flat.unsqueeze(0)).item()
            
            # 相对误差
            relative_error = (torch.norm(f1 - f2) / torch.norm(f1)).item()
            
            # 最大绝对差异
            max_abs_diff = torch.max(torch.abs(f1 - f2)).item()
            
            # 统计信息
            f1_stats = {
                'mean': f1.mean().item(),
                'std': f1.std().item(),
                'min': f1.min().item(),
                'max': f1.max().item(),
            }
            
            f2_stats = {
                'mean': f2.mean().item(),
                'std': f2.std().item(),
                'min': f2.min().item(),
                'max': f2.max().item(),
            }
            
            results[key] = {
                'shape': f1.shape,
                'mse': mse,
                'mae': mae,
                'cosine_similarity': cosine_sim,
                'relative_error': relative_error,
                'max_abs_diff': max_abs_diff,
                f'{names[0]}_stats': f1_stats,
                f'{names[1]}_stats': f2_stats,
            }
    
    return results

def print_comparison_results(results, names):
    """打印对比结果"""
    print(f"\n{'='*80}")
    print(f"特征对比结果: {names[0]} vs {names[1]}")
    print(f"{'='*80}")
    
    for key, metrics in results.items():
        print(f"\n🔍 {key.upper()}:")
        print(f"  形状: {metrics['shape']}")
        print(f"  MSE: {metrics['mse']:.2e}")
        print(f"  MAE: {metrics['mae']:.2e}")
        print(f"  余弦相似性: {metrics['cosine_similarity']:.6f}")
        print(f"  相对误差: {metrics['relative_error']:.2e}")
        print(f"  最大绝对差异: {metrics['max_abs_diff']:.2e}")
        
        # 统计对比
        stats1 = metrics[f'{names[0]}_stats']
        stats2 = metrics[f'{names[1]}_stats']
        
        print(f"  统计对比:")
        for stat_name in ['mean', 'std', 'min', 'max']:
            val1 = stats1[stat_name]
            val2 = stats2[stat_name]
            diff = abs(val1 - val2)
            print(f"    {stat_name}: {val1:.4f} vs {val2:.4f} (差异: {diff:.2e})")

def main():
    """主测试函数"""
    print("🔍 EoMT vs 原版DINOv3特征对比测试")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    original_dinov3 = load_original_dinov3().to(device).eval()
    eomt_model = create_eomt_model().to(device).eval()
    
    print("✓ 原版DINOv3加载完成")
    print("✓ EoMT模型加载完成")
    
    # 2. 准备测试输入
    print("\n2. 准备测试输入...")
    batch_size = 2
    test_inputs = [
        torch.randn(batch_size, 3, 224, 224).to(device),  # 随机输入
        torch.zeros(batch_size, 3, 224, 224).to(device),  # 全零输入
        torch.ones(batch_size, 3, 224, 224).to(device),   # 全一输入
    ]
    
    input_names = ['随机输入', '全零输入', '全一输入']
    
    # 3. 对每个输入进行测试
    for i, (x, input_name) in enumerate(zip(test_inputs, input_names)):
        print(f"\n{'='*20} {input_name} {'='*20}")
        
        # 提取特征
        print("提取原版DINOv3特征...")
        original_features = extract_dinov3_features(original_dinov3, x)
        
        print("提取EoMT特征...")
        eomt_features = extract_eomt_features(eomt_model, x)
        
        # 计算差异
        print("计算特征差异...")
        differences = compute_feature_differences(
            original_features, 
            eomt_features, 
            ['原版DINOv3', 'EoMT实现']
        )
        
        # 打印结果
        print_comparison_results(differences, ['原版DINOv3', 'EoMT实现'])
        
        # 简要总结
        print(f"\n📊 {input_name} 总结:")
        avg_cosine_sim = np.mean([d['cosine_similarity'] for d in differences.values()])
        avg_relative_error = np.mean([d['relative_error'] for d in differences.values()])
        
        print(f"  平均余弦相似性: {avg_cosine_sim:.6f}")
        print(f"  平均相对误差: {avg_relative_error:.2e}")
        
        if avg_cosine_sim > 0.999:
            print("  ✅ 特征高度一致!")
        elif avg_cosine_sim > 0.99:
            print("  ⚠️ 特征基本一致，有小差异")
        else:
            print("  ❌ 特征差异较大")
    
    print(f"\n{'='*60}")
    print("🎉 特征对比测试完成!")
    print("="*60)

if __name__ == "__main__":
    main()
