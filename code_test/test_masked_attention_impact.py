#!/usr/bin/env python3
"""
专门测试Masked Attention对特征的影响
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, 'models/dinov3')

from models.vit import ViT
from models.eomt import EoMT
from torchvision import transforms
from PIL import Image
import numpy as np

def load_eomt_model():
    """加载EoMT模型"""
    encoder = ViT(img_size=(224, 224))
    eomt = EoMT(
        encoder=encoder,
        num_classes=80,
        num_q=100,
        num_blocks=4,  # 最后4层插入query
        masked_attn_enabled=True  # 先设为True，后面会动态修改
    )
    return eomt

def create_test_input():
    """创建测试输入"""
    torch.manual_seed(42)  # 固定随机种子
    return torch.randn(1, 3, 224, 224)

def extract_intermediate_features(model, x, masked_attn_enabled):
    """提取中间层特征"""
    model.masked_attn_enabled = masked_attn_enabled
    model.eval()
    
    with torch.no_grad():
        # 手动执行EoMT的forward，保存中间状态
        # 预处理
        x_norm = (x - model.encoder.pixel_mean) / model.encoder.pixel_std
        
        # Token准备
        tokens, (H, W) = model.encoder.backbone.prepare_tokens_with_masks(x_norm)
        
        # 保存每层的状态
        layer_states = []
        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []
        
        total_blocks = len(model.encoder.backbone.blocks)
        query_start_layer = total_blocks - model.num_blocks
        
        for i, block in enumerate(model.encoder.backbone.blocks):
            # 插入query tokens
            if i == query_start_layer:
                tokens = torch.cat(
                    (model.q.weight[None, :, :].expand(tokens.shape[0], -1, -1), tokens), dim=1
                )
            
            # RoPE计算
            if model.encoder.backbone.rope_embed is not None:
                rope_or_rope_list = model.encoder.backbone.rope_embed(H=H, W=W)
            else:
                rope_or_rope_list = None
            
            # Masked attention逻辑
            if (
                masked_attn_enabled
                and i >= query_start_layer
            ):
                # 使用DINOv3的norm层
                if hasattr(model.encoder.backbone, 'norm'):
                    normed_x = model.encoder.backbone.norm(tokens)
                else:
                    normed_x = tokens
                    
                mask_logits, class_logits = model._predict(normed_x)
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                # 创建attention mask
                attn_mask = torch.ones(
                    tokens.shape[0],
                    tokens.shape[1],
                    tokens.shape[1],
                    dtype=torch.bool,
                    device=tokens.device,
                )
                
                # 使用实际的H, W进行插值
                interpolated = F.interpolate(
                    mask_logits,
                    size=(H, W),
                    mode="bilinear",
                )
                interpolated = interpolated.view(
                    interpolated.size(0), interpolated.size(1), -1
                )
                attn_mask[
                    :,
                    : model.num_q,
                    model.num_q + model.num_prefix_tokens :,
                ] = (
                    interpolated > 0
                )
                attn_mask = model._disable_attn_mask(
                    attn_mask,
                    model.attn_mask_probs[i - query_start_layer],
                )

                # 使用自定义的attention来处理mask
                tokens = tokens + block.ls1(
                    model._attn(block.attn, block.norm1(tokens), attn_mask, rope_or_rope_list)
                )
                tokens = tokens + block.ls2(block.mlp(block.norm2(tokens)))
            else:
                # 没有mask时，直接使用DINOv3的block forward
                tokens = block(tokens, rope_or_rope_list)
            
            # 保存当前层状态
            layer_states.append({
                'layer': i,
                'tokens': tokens.clone(),
                'has_queries': i >= query_start_layer,
                'masked_attn_active': masked_attn_enabled and i >= query_start_layer,
                'attn_mask': attn_mask.clone() if attn_mask is not None else None,
            })
        
        return layer_states, mask_logits_per_layer, class_logits_per_layer

def compare_masked_attention_effects(model, x):
    """对比Masked Attention的影响"""
    print("\n🔍 Masked Attention影响分析:")
    
    # 1. 无Masked Attention
    print("\n  📋 无Masked Attention:")
    states_no_mask, _, _ = extract_intermediate_features(model, x, masked_attn_enabled=False)
    
    # 2. 有Masked Attention
    print("  📋 有Masked Attention:")
    states_with_mask, mask_logits, class_logits = extract_intermediate_features(model, x, masked_attn_enabled=True)
    
    print(f"    输出层数: {len(mask_logits)}")
    
    # 3. 逐层对比
    print(f"\n  📊 逐层差异分析:")
    
    query_start_layer = len(model.encoder.backbone.blocks) - model.num_blocks
    
    for i in range(len(states_no_mask)):
        state_no_mask = states_no_mask[i]
        state_with_mask = states_with_mask[i]
        
        tokens_no_mask = state_no_mask['tokens']
        tokens_with_mask = state_with_mask['tokens']
        
        # 计算差异
        if tokens_no_mask.shape == tokens_with_mask.shape:
            mse = F.mse_loss(tokens_no_mask, tokens_with_mask).item()
            cosine_sim = F.cosine_similarity(
                tokens_no_mask.flatten(), 
                tokens_with_mask.flatten(), 
                dim=0
            ).item()
            
            # 分析差异程度
            if mse < 1e-6:
                status = "✅ 完全一致"
            elif cosine_sim > 0.99:
                status = "🟡 微小差异"
            elif cosine_sim > 0.9:
                status = "🟠 中等差异"
            else:
                status = "🔴 显著差异"
            
            if i < query_start_layer:
                # Query插入前
                print(f"    Layer {i:2d}: {status} MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f}")
            elif i == query_start_layer:
                # Query插入层
                print(f"    Layer {i:2d}: {status} MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f} [Query插入层]")
            else:
                # Query插入后
                masked_active = state_with_mask['masked_attn_active']
                mask_status = "有Mask" if masked_active else "无Mask"
                print(f"    Layer {i:2d}: {status} MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f} [{mask_status}]")
        else:
            print(f"    Layer {i:2d}: ❌ 形状不匹配 - {tokens_no_mask.shape} vs {tokens_with_mask.shape}")
    
    # 4. 分析Query tokens的变化
    print(f"\n  🎯 Query Tokens分析:")
    
    for i in range(query_start_layer, len(states_no_mask)):
        state_no_mask = states_no_mask[i]
        state_with_mask = states_with_mask[i]
        
        # 提取query tokens部分
        query_no_mask = state_no_mask['tokens'][:, :model.num_q, :]
        query_with_mask = state_with_mask['tokens'][:, :model.num_q, :]
        
        mse = F.mse_loss(query_no_mask, query_with_mask).item()
        cosine_sim = F.cosine_similarity(
            query_no_mask.flatten(), 
            query_with_mask.flatten(), 
            dim=0
        ).item()
        
        masked_active = state_with_mask['masked_attn_active']
        mask_status = "有Mask" if masked_active else "无Mask"
        
        print(f"    Layer {i:2d} Query: MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f} [{mask_status}]")
    
    # 5. 分析Backbone tokens的变化
    print(f"\n  🏗️ Backbone Tokens分析:")
    
    for i in range(query_start_layer, len(states_no_mask)):
        state_no_mask = states_no_mask[i]
        state_with_mask = states_with_mask[i]
        
        # 提取backbone tokens部分
        backbone_no_mask = state_no_mask['tokens'][:, model.num_q:, :]
        backbone_with_mask = state_with_mask['tokens'][:, model.num_q:, :]
        
        mse = F.mse_loss(backbone_no_mask, backbone_with_mask).item()
        cosine_sim = F.cosine_similarity(
            backbone_no_mask.flatten(), 
            backbone_with_mask.flatten(), 
            dim=0
        ).item()
        
        masked_active = state_with_mask['masked_attn_active']
        mask_status = "有Mask" if masked_active else "无Mask"
        
        print(f"    Layer {i:2d} Backbone: MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f} [{mask_status}]")

def main():
    """主函数"""
    print("🔍 Masked Attention影响详细分析")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    model = load_eomt_model().to(device).eval()
    
    print(f"  EoMT配置: num_blocks={model.num_blocks}, num_q={model.num_q}")
    print(f"  Query插入位置: 第{len(model.encoder.backbone.blocks) - model.num_blocks}层开始")
    
    # 2. 创建测试输入
    print("\n2. 创建测试输入...")
    x = create_test_input().to(device)
    print(f"测试输入: shape={x.shape}")
    
    # 3. 对比分析
    compare_masked_attention_effects(model, x)
    
    print(f"\n{'='*60}")
    print("📋 关键洞察:")
    print("1. 前20层：无差异（还没有query tokens）")
    print("2. 第20层：插入query后的基础差异")
    print("3. 第21-23层：Masked Attention的额外影响")
    print("4. Query tokens vs Backbone tokens：不同的受影响程度")
    print("="*60)

if __name__ == "__main__":
    main()
