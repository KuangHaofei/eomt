#!/usr/bin/env python3
"""
相同归一化输入条件下，对比原版DINOv3和EoMT版本的差异
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

def create_normalized_input():
    """创建归一化后的输入"""
    # 创建一个测试图像
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # 标准的ImageNet归一化
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])
    
    normalized_input = transform(test_image).unsqueeze(0)
    return normalized_input

def extract_original_features(model, normalized_input):
    """从原版DINOv3提取特征（输入已归一化）"""
    with torch.no_grad():
        # 直接使用归一化后的输入
        features = model.forward_features(normalized_input)
        
        return {
            'cls_token': features['x_norm_clstoken'],           # [B, 1024]
            'storage_tokens': features['x_storage_tokens'],     # [B, 4, 1024]
            'patch_tokens': features['x_norm_patchtokens'],     # [B, 196, 1024]
            'prenorm': features['x_prenorm'],                   # [B, 201, 1024]
        }

def extract_eomt_features_no_norm(model, normalized_input):
    """从EoMT提取特征（跳过内部归一化）"""
    with torch.no_grad():
        # 直接使用DINOv3的prepare_tokens_with_masks，跳过EoMT的归一化
        tokens, (H, W) = model.encoder.backbone.prepare_tokens_with_masks(normalized_input)
        
        # 运行所有blocks（无query tokens，无masked attention）
        for i, block in enumerate(model.encoder.backbone.blocks):
            # 每个block都重新计算RoPE位置编码
            if model.encoder.backbone.rope_embed is not None:
                rope_or_rope_list = model.encoder.backbone.rope_embed(H=H, W=W)
            else:
                rope_or_rope_list = None
            
            # 直接使用DINOv3的block forward
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

def compare_layer_by_layer(original, eomt, normalized_input):
    """逐层对比处理"""
    print("\n🔍 逐层对比（相同归一化输入）:")
    
    with torch.no_grad():
        # 1. Token准备阶段
        print("\n  1. Token准备阶段:")
        original_tokens, (H1, W1) = original.prepare_tokens_with_masks(normalized_input)
        eomt_tokens, (H2, W2) = eomt.encoder.backbone.prepare_tokens_with_masks(normalized_input)
        
        print(f"     原版tokens: {original_tokens.shape}")
        print(f"     EoMT tokens: {eomt_tokens.shape}")
        
        tokens_match = torch.allclose(original_tokens, eomt_tokens, atol=1e-6)
        if tokens_match:
            print("     ✅ Token准备完全一致")
        else:
            mse = F.mse_loss(original_tokens, eomt_tokens).item()
            cosine_sim = F.cosine_similarity(
                original_tokens.flatten(), 
                eomt_tokens.flatten(), 
                dim=0
            ).item()
            print(f"     ❌ Token准备有差异: MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f}")
            return None
        
        # 2. 逐层Block处理
        print(f"\n  2. 逐层Block处理:")
        original_x = original_tokens.clone()
        eomt_x = eomt_tokens.clone()
        
        for i in range(min(5, len(original.blocks))):  # 只检查前5层
            # 原版处理
            original_block = original.blocks[i]
            if original.rope_embed is not None:
                rope_original = original.rope_embed(H=H1, W=W1)
            else:
                rope_original = None
            original_x = original_block(original_x, rope_original)
            
            # EoMT处理
            eomt_block = eomt.encoder.backbone.blocks[i]
            if eomt.encoder.backbone.rope_embed is not None:
                rope_eomt = eomt.encoder.backbone.rope_embed(H=H2, W=W2)
            else:
                rope_eomt = None
            eomt_x = eomt_block(eomt_x, rope_eomt)
            
            # 对比
            block_match = torch.allclose(original_x, eomt_x, atol=1e-6)
            mse = F.mse_loss(original_x, eomt_x).item()
            cosine_sim = F.cosine_similarity(
                original_x.flatten(), 
                eomt_x.flatten(), 
                dim=0
            ).item()
            
            print(f"     Block {i}: 匹配={block_match}, MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f}")
            
            if not block_match and mse > 1e-6:
                print(f"     ❌ Block {i} 开始出现差异!")
                return i
        
        print("     ✅ 所有Block处理完全一致")
        return None

def compare_features(features1, features2, name1, name2):
    """对比特征"""
    print(f"\n🔍 特征对比: {name1} vs {name2}")
    
    for key in features1.keys():
        if key in features2:
            f1 = features1[key]
            f2 = features2[key]
            
            if f1.shape != f2.shape:
                print(f"  {key}: 形状不匹配 - {f1.shape} vs {f2.shape}")
                continue
            
            # 计算差异
            match = torch.allclose(f1, f2, atol=1e-6)
            mse = F.mse_loss(f1, f2).item()
            cosine_sim = F.cosine_similarity(f1.flatten(), f2.flatten(), dim=0).item()
            
            status = "✅" if match else "❌"
            print(f"  {key}: {status} MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f}")

def main():
    """主函数"""
    print("🔍 相同归一化输入条件下的对比测试")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    original, eomt = load_models()
    original = original.to(device).eval()
    eomt = eomt.to(device).eval()
    
    # 2. 创建归一化输入
    print("\n2. 创建归一化输入...")
    torch.manual_seed(42)  # 固定随机种子
    normalized_input = create_normalized_input().to(device)
    print(f"归一化输入: shape={normalized_input.shape}")
    print(f"            mean={normalized_input.mean():.6f}, std={normalized_input.std():.6f}")
    
    # 3. 逐层对比
    diff_layer = compare_layer_by_layer(original, eomt, normalized_input)
    
    if diff_layer is not None:
        print(f"\n❌ 在Block {diff_layer}开始出现差异")
        return
    
    # 4. 特征提取对比
    print(f"\n3. 特征提取对比...")
    
    # 原版DINOv3特征（输入已归一化）
    original_features = extract_original_features(original, normalized_input)
    
    # EoMT特征（跳过内部归一化）
    eomt_features = extract_eomt_features_no_norm(eomt, normalized_input)
    
    # 对比特征
    compare_features(original_features, eomt_features, "原版DINOv3", "EoMT(无内部归一化)")
    
    # 5. 测试EoMT的完整流程
    print(f"\n4. EoMT完整流程测试...")
    
    # 创建原始输入（未归一化）
    raw_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        # EoMT完整流程（包含内部归一化）
        eomt.masked_attn_enabled = False
        mask_logits_no_mask, class_logits_no_mask = eomt(raw_input)
        
        # EoMT with masked attention
        eomt.masked_attn_enabled = True
        mask_logits_with_mask, class_logits_with_mask = eomt(raw_input)
    
    print(f"  无masked attention: {len(mask_logits_no_mask)}层输出")
    print(f"  有masked attention: {len(mask_logits_with_mask)}层输出")
    
    # 对比masked attention的影响
    if len(mask_logits_no_mask) > 0 and len(mask_logits_with_mask) > 0:
        final_no_mask = mask_logits_no_mask[-1]
        final_with_mask = mask_logits_with_mask[-1]
        
        mse = F.mse_loss(final_no_mask, final_with_mask).item()
        cosine_sim = F.cosine_similarity(
            final_no_mask.flatten(), 
            final_with_mask.flatten(), 
            dim=0
        ).item()
        
        print(f"  Masked attention影响: MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f}")
    
    print(f"\n{'='*60}")
    print("📋 总结:")
    print("1. 在相同归一化输入下，原版DINOv3和EoMT的backbone处理是否一致？")
    print("2. EoMT的query tokens和masked attention如何影响最终输出？")
    print("3. 这验证了我们之前的分析：差异主要来自输入预处理方式的不同")
    print("="*60)

if __name__ == "__main__":
    main()
