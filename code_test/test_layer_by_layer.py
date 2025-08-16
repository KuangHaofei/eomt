#!/usr/bin/env python3
"""
逐层对比EoMT和原版DINOv3的特征
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, 'models/dinov3')

from models.vit import ViT

def load_models():
    """加载模型"""
    # 原版DINOv3
    repo_dir = 'models/dinov3'
    weights_path = 'models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    original = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', weights=weights_path)
    
    # EoMT的encoder
    encoder = ViT(img_size=(224, 224))
    
    return original, encoder

def compare_initial_processing(original_model, encoder, x):
    """对比初始处理步骤"""
    print("\n🔍 对比初始处理步骤:")
    
    # 原版DINOv3
    with torch.no_grad():
        original_tokens, (H1, W1) = original_model.prepare_tokens_with_masks(x)
    
    # EoMT实现
    with torch.no_grad():
        x_norm = (x - encoder.pixel_mean) / encoder.pixel_std
        eomt_tokens, (H2, W2) = encoder.backbone.prepare_tokens_with_masks(x_norm)
    
    print(f"  原版tokens: {original_tokens.shape}")
    print(f"  EoMT tokens: {eomt_tokens.shape}")
    print(f"  H,W: ({H1},{W1}) vs ({H2},{W2})")
    
    # 检查tokens一致性
    tokens_match = torch.allclose(original_tokens, eomt_tokens, atol=1e-6)
    print(f"  Tokens匹配: {tokens_match}")
    
    if not tokens_match:
        mse = F.mse_loss(original_tokens, eomt_tokens).item()
        cosine_sim = F.cosine_similarity(
            original_tokens.flatten(), 
            eomt_tokens.flatten(), 
            dim=0
        ).item()
        print(f"  MSE: {mse:.2e}")
        print(f"  余弦相似性: {cosine_sim:.6f}")
    
    return original_tokens, eomt_tokens, (H1, W1)

def compare_block_by_block(original_model, encoder, original_tokens, eomt_tokens, hw):
    """逐个block对比"""
    print(f"\n🔍 逐层block对比:")
    H, W = hw
    
    original_x = original_tokens.clone()
    eomt_x = eomt_tokens.clone()
    
    for i in range(min(5, len(original_model.blocks))):  # 只比较前5层
        print(f"\n  Block {i}:")
        
        # 原版DINOv3处理
        with torch.no_grad():
            original_block = original_model.blocks[i]
            if original_model.rope_embed is not None:
                rope_original = original_model.rope_embed(H=H, W=W)
            else:
                rope_original = None
            original_x = original_block(original_x, rope_original)
        
        # EoMT处理
        with torch.no_grad():
            eomt_block = encoder.backbone.blocks[i]
            if encoder.backbone.rope_embed is not None:
                rope_eomt = encoder.backbone.rope_embed(H=H, W=W)
            else:
                rope_eomt = None
            eomt_x = eomt_block(eomt_x, rope_eomt)
        
        # 对比结果
        block_match = torch.allclose(original_x, eomt_x, atol=1e-6)
        mse = F.mse_loss(original_x, eomt_x).item()
        cosine_sim = F.cosine_similarity(
            original_x.flatten(), 
            eomt_x.flatten(), 
            dim=0
        ).item()
        
        print(f"    形状: {original_x.shape} vs {eomt_x.shape}")
        print(f"    完全匹配: {block_match}")
        print(f"    MSE: {mse:.2e}")
        print(f"    余弦相似性: {cosine_sim:.6f}")
        
        if mse > 1e-6:
            print(f"    ⚠️ Block {i} 有差异!")
            
            # 检查RoPE是否一致
            if rope_original is not None and rope_eomt is not None:
                rope_match = (
                    torch.allclose(rope_original[0], rope_eomt[0], atol=1e-6) and
                    torch.allclose(rope_original[1], rope_eomt[1], atol=1e-6)
                )
                print(f"    RoPE匹配: {rope_match}")
            
            break  # 如果发现差异，停止继续比较
    
    return original_x, eomt_x

def compare_final_norm(original_model, encoder, original_x, eomt_x):
    """对比最终归一化"""
    print(f"\n🔍 对比最终归一化:")
    
    with torch.no_grad():
        original_norm = original_model.norm(original_x)
        eomt_norm = encoder.backbone.norm(eomt_x)
    
    norm_match = torch.allclose(original_norm, eomt_norm, atol=1e-6)
    mse = F.mse_loss(original_norm, eomt_norm).item()
    cosine_sim = F.cosine_similarity(
        original_norm.flatten(), 
        eomt_norm.flatten(), 
        dim=0
    ).item()
    
    print(f"  形状: {original_norm.shape} vs {eomt_norm.shape}")
    print(f"  完全匹配: {norm_match}")
    print(f"  MSE: {mse:.2e}")
    print(f"  余弦相似性: {cosine_sim:.6f}")

def main():
    """主函数"""
    print("🔍 EoMT vs 原版DINOv3 逐层对比测试")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\\n1. 加载模型...")
    original_model, encoder = load_models()
    original_model = original_model.to(device).eval()
    encoder = encoder.to(device).eval()
    
    # 准备测试输入
    print("\\n2. 准备测试输入...")
    x = torch.randn(1, 3, 224, 224).to(device)
    print(f"输入形状: {x.shape}")
    
    # 对比初始处理
    original_tokens, eomt_tokens, hw = compare_initial_processing(original_model, encoder, x)
    
    # 逐层对比
    original_final, eomt_final = compare_block_by_block(
        original_model, encoder, original_tokens, eomt_tokens, hw
    )
    
    # 对比最终归一化
    compare_final_norm(original_model, encoder, original_final, eomt_final)
    
    print(f"\\n{'='*60}")
    print("🎉 逐层对比测试完成!")
    print("="*60)

if __name__ == "__main__":
    main()
