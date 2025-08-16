#!/usr/bin/env python3
"""
精确定位EoMT和原版DINOv3特征差异的起始层
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, 'models/dinov3')

from models.vit import ViT
from models.eomt import EoMT

def load_models():
    """加载模型"""
    # 原版DINOv3
    repo_dir = 'models/dinov3'
    weights_path = 'models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    original = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', weights=weights_path)
    
    # EoMT模型
    encoder = ViT(img_size=(224, 224))
    eomt = EoMT(
        encoder=encoder,
        num_classes=80,
        num_q=100,
        num_blocks=4,
        masked_attn_enabled=False  # 先关闭masked attention
    )
    
    return original, eomt

def compare_preprocessing(original, eomt, x):
    """对比预处理步骤"""
    print("\n🔍 步骤1: 预处理对比")
    
    with torch.no_grad():
        # 原版DINOv3: 直接使用prepare_tokens_with_masks
        original_tokens, (H1, W1) = original.prepare_tokens_with_masks(x)
        
        # EoMT: 模拟forward的前几步
        x_norm = (x - eomt.encoder.pixel_mean) / eomt.encoder.pixel_std
        eomt_tokens, (H2, W2) = eomt.encoder.backbone.prepare_tokens_with_masks(x_norm)
    
    print(f"  原版tokens: {original_tokens.shape}")
    print(f"  EoMT tokens: {eomt_tokens.shape}")
    print(f"  H,W: ({H1},{W1}) vs ({H2},{W2})")
    
    # 检查差异
    tokens_match = torch.allclose(original_tokens, eomt_tokens, atol=1e-6)
    if not tokens_match:
        mse = F.mse_loss(original_tokens, eomt_tokens).item()
        cosine_sim = F.cosine_similarity(
            original_tokens.flatten(), 
            eomt_tokens.flatten(), 
            dim=0
        ).item()
        print(f"  ❌ 预处理就有差异!")
        print(f"    MSE: {mse:.2e}")
        print(f"    余弦相似性: {cosine_sim:.6f}")
        return None, None, None  # 预处理就不同，无法继续比较
    else:
        print(f"  ✅ 预处理完全一致")
    
    return original_tokens, eomt_tokens, (H1, W1)

def compare_blocks_without_queries(original, eomt, original_tokens, eomt_tokens, hw):
    """对比没有query tokens的block处理"""
    print(f"\n🔍 步骤2: 逐层Block对比（无query tokens）")
    H, W = hw
    
    original_x = original_tokens.clone()
    eomt_x = eomt_tokens.clone()
    
    total_blocks = len(original.blocks)
    
    for i in range(total_blocks):
        print(f"\n  Block {i}:")
        
        # 原版DINOv3处理
        with torch.no_grad():
            original_block = original.blocks[i]
            if original.rope_embed is not None:
                rope_original = original.rope_embed(H=H, W=W)
            else:
                rope_original = None
            original_x = original_block(original_x, rope_original)
        
        # EoMT处理（模拟无query tokens的情况）
        with torch.no_grad():
            eomt_block = eomt.encoder.backbone.blocks[i]
            if eomt.encoder.backbone.rope_embed is not None:
                rope_eomt = eomt.encoder.backbone.rope_embed(H=H, W=W)
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
        
        print(f"    完全匹配: {block_match}")
        print(f"    MSE: {mse:.2e}")
        print(f"    余弦相似性: {cosine_sim:.6f}")
        
        if not block_match and mse > 1e-6:
            print(f"    ❌ Block {i} 开始出现差异!")
            return i, original_x, eomt_x
    
    print(f"  ✅ 所有Block处理完全一致")
    return None, original_x, eomt_x

def compare_with_queries(original, eomt, x):
    """对比加入query tokens后的处理"""
    print(f"\n🔍 步骤3: 加入query tokens后的对比")
    
    # 使用EoMT的完整forward（无masked attention）
    eomt.masked_attn_enabled = False
    eomt.eval()
    
    with torch.no_grad():
        # 原版DINOv3特征
        original_features = original.forward_features(x)
        
        # EoMT特征（无masked attention）
        mask_logits, class_logits = eomt(x)
    
    print(f"  原版特征形状: {original_features['x_prenorm'].shape}")
    print(f"  EoMT输出: mask_logits={mask_logits[0].shape}, class_logits={class_logits[0].shape}")
    print(f"  ✅ 加入query tokens后，两者处理方式不同，这是正常的")

def compare_with_masked_attention(eomt, x):
    """对比开启masked attention的影响"""
    print(f"\n🔍 步骤4: Masked Attention的影响")
    
    eomt.eval()
    
    with torch.no_grad():
        # 无masked attention
        eomt.masked_attn_enabled = False
        mask_logits_no_mask, class_logits_no_mask = eomt(x)
        
        # 有masked attention
        eomt.masked_attn_enabled = True
        mask_logits_with_mask, class_logits_with_mask = eomt(x)
    
    print(f"  无mask输出层数: {len(mask_logits_no_mask)}")
    print(f"  有mask输出层数: {len(mask_logits_with_mask)}")
    
    # 对比最终输出
    if len(mask_logits_no_mask) > 0 and len(mask_logits_with_mask) > 0:
        final_no_mask = mask_logits_no_mask[-1]
        final_with_mask = mask_logits_with_mask[-1]
        
        mse = F.mse_loss(final_no_mask, final_with_mask).item()
        cosine_sim = F.cosine_similarity(
            final_no_mask.flatten(), 
            final_with_mask.flatten(), 
            dim=0
        ).item()
        
        print(f"  最终输出MSE: {mse:.2e}")
        print(f"  最终输出余弦相似性: {cosine_sim:.6f}")
        
        if mse > 1e-3:
            print(f"  ❌ Masked attention显著改变了输出")
        else:
            print(f"  ✅ Masked attention影响较小")

def main():
    """主函数"""
    print("🔍 EoMT vs 原版DINOv3 差异定位分析")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n1. 加载模型...")
    original, eomt = load_models()
    original = original.to(device).eval()
    eomt = eomt.to(device).eval()
    
    # 准备测试输入
    print("\n2. 准备测试输入...")
    torch.manual_seed(42)  # 固定随机种子
    x = torch.randn(1, 3, 224, 224).to(device)
    print(f"输入形状: {x.shape}")
    
    # 步骤1: 对比预处理
    original_tokens, eomt_tokens, hw = compare_preprocessing(original, eomt, x)
    
    if original_tokens is None:
        print("\n❌ 预处理阶段就有差异，无法继续分析")
        return
    
    # 步骤2: 对比block处理（无query tokens）
    diff_block, original_final, eomt_final = compare_blocks_without_queries(
        original, eomt, original_tokens, eomt_tokens, hw
    )
    
    if diff_block is not None:
        print(f"\n❌ 在Block {diff_block}开始出现差异")
    else:
        print(f"\n✅ 在没有query tokens的情况下，所有处理完全一致")
    
    # 步骤3: 对比加入query tokens
    compare_with_queries(original, eomt, x)
    
    # 步骤4: 对比masked attention的影响
    compare_with_masked_attention(eomt, x)
    
    print(f"\n{'='*60}")
    print("📋 总结:")
    print("1. 预处理阶段是否一致？")
    print("2. 纯DINOv3 block处理是否一致？")
    print("3. 加入query tokens后的差异是预期的")
    print("4. Masked attention会进一步改变特征")
    print("="*60)

if __name__ == "__main__":
    main()
