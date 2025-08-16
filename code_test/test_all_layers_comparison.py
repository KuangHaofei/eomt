#!/usr/bin/env python3
"""
完整测试所有24层，特别关注EoMT插入query tokens的影响
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
        num_blocks=4,  # 最后4层插入query
        masked_attn_enabled=False  # 先关闭masked attention
    )
    
    return original, eomt

def create_normalized_input():
    """创建归一化后的输入"""
    torch.manual_seed(42)  # 固定随机种子
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])
    
    return transform(test_image).unsqueeze(0)

def simulate_eomt_processing(eomt, normalized_input):
    """模拟EoMT的完整处理流程"""
    with torch.no_grad():
        # 1. Token准备（跳过内部归一化）
        x, (H, W) = eomt.encoder.backbone.prepare_tokens_with_masks(normalized_input)
        
        total_blocks = len(eomt.encoder.backbone.blocks)
        query_start_layer = total_blocks - eomt.num_blocks  # 第20层开始插入query
        
        print(f"  总共{total_blocks}层，第{query_start_layer}层开始插入query tokens")
        
        # 2. 逐层处理
        for i, block in enumerate(eomt.encoder.backbone.blocks):
            print(f"\n    Layer {i}:")
            print(f"      输入shape: {x.shape}")
            
            # 在指定层插入query tokens
            if i == query_start_layer:
                print(f"      🔥 插入{eomt.num_q}个query tokens")
                query_tokens = eomt.q.weight[None, :, :].expand(x.shape[0], -1, -1)
                x = torch.cat((query_tokens, x), dim=1)
                print(f"      插入后shape: {x.shape}")
            
            # RoPE计算
            if eomt.encoder.backbone.rope_embed is not None:
                rope_or_rope_list = eomt.encoder.backbone.rope_embed(H=H, W=W)
            else:
                rope_or_rope_list = None
            
            # Block处理
            x = block(x, rope_or_rope_list)
            print(f"      输出shape: {x.shape}")
        
        return x

def compare_original_vs_eomt(original, eomt, normalized_input):
    """对比原版DINOv3和EoMT的逐层处理"""
    print("\n🔍 逐层详细对比:")
    
    with torch.no_grad():
        # 原版DINOv3处理
        print("\n  📋 原版DINOv3处理:")
        original_tokens, (H1, W1) = original.prepare_tokens_with_masks(normalized_input)
        print(f"    初始tokens: {original_tokens.shape}")
        
        original_x = original_tokens.clone()
        original_states = []  # 保存每层的状态
        
        for i, block in enumerate(original.blocks):
            if original.rope_embed is not None:
                rope = original.rope_embed(H=H1, W=W1)
            else:
                rope = None
            
            original_x = block(original_x, rope)
            original_states.append(original_x.clone())
            
            if i < 5 or i >= 20:  # 只打印前5层和后5层
                print(f"    Layer {i}: {original_x.shape}")
        
        # EoMT处理
        print(f"\n  📋 EoMT处理:")
        eomt_tokens, (H2, W2) = eomt.encoder.backbone.prepare_tokens_with_masks(normalized_input)
        print(f"    初始tokens: {eomt_tokens.shape}")
        
        eomt_x = eomt_tokens.clone()
        eomt_states = []
        
        total_blocks = len(eomt.encoder.backbone.blocks)
        query_start_layer = total_blocks - eomt.num_blocks
        
        for i, block in enumerate(eomt.encoder.backbone.blocks):
            # 插入query tokens
            if i == query_start_layer:
                print(f"    🔥 Layer {i}: 插入{eomt.num_q}个query tokens")
                query_tokens = eomt.q.weight[None, :, :].expand(eomt_x.shape[0], -1, -1)
                eomt_x = torch.cat((query_tokens, eomt_x), dim=1)
                print(f"    插入后: {eomt_x.shape}")
            
            # RoPE和Block处理
            if eomt.encoder.backbone.rope_embed is not None:
                rope = eomt.encoder.backbone.rope_embed(H=H2, W=W2)
            else:
                rope = None
            
            eomt_x = block(eomt_x, rope)
            eomt_states.append(eomt_x.clone())
            
            if i < 5 or i >= 20:
                print(f"    Layer {i}: {eomt_x.shape}")
        
        # 对比分析
        print(f"\n  📊 层级对比分析:")
        
        for i in range(total_blocks):
            original_state = original_states[i]
            eomt_state = eomt_states[i]
            
            if i < query_start_layer:
                # Query插入前：应该完全一致
                if original_state.shape == eomt_state.shape:
                    match = torch.allclose(original_state, eomt_state, atol=1e-6)
                    mse = F.mse_loss(original_state, eomt_state).item()
                    cosine_sim = F.cosine_similarity(
                        original_state.flatten(), 
                        eomt_state.flatten(), 
                        dim=0
                    ).item()
                    
                    status = "✅" if match else "❌"
                    print(f"    Layer {i}: {status} MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f}")
                    
                    if not match:
                        print(f"      ⚠️ 在插入query之前就有差异！")
                        return i
                else:
                    print(f"    Layer {i}: ❌ 形状不匹配 - {original_state.shape} vs {eomt_state.shape}")
                    return i
            else:
                # Query插入后：形状会不同，只对比backbone部分
                if i == query_start_layer:
                    print(f"    Layer {i}: 🔥 Query插入层 - 形状变化是预期的")
                    print(f"      原版: {original_state.shape}")
                    print(f"      EoMT: {eomt_state.shape}")
                    
                    # 比较backbone部分（跳过query tokens）
                    eomt_backbone_part = eomt_state[:, eomt.num_q:, :]  # 跳过前100个query tokens
                    
                    if original_state.shape == eomt_backbone_part.shape:
                        match = torch.allclose(original_state, eomt_backbone_part, atol=1e-6)
                        mse = F.mse_loss(original_state, eomt_backbone_part).item()
                        cosine_sim = F.cosine_similarity(
                            original_state.flatten(), 
                            eomt_backbone_part.flatten(), 
                            dim=0
                        ).item()
                        
                        status = "✅" if match else "❌"
                        print(f"      Backbone部分: {status} MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f}")
                        
                        if not match:
                            print(f"      ⚠️ 插入query后backbone部分有差异！")
                            return i
                    else:
                        print(f"      ❌ Backbone部分形状不匹配")
                        return i
                else:
                    # 后续层：继续比较backbone部分
                    eomt_backbone_part = eomt_state[:, eomt.num_q:, :]
                    
                    if original_state.shape == eomt_backbone_part.shape:
                        match = torch.allclose(original_state, eomt_backbone_part, atol=1e-6)
                        mse = F.mse_loss(original_state, eomt_backbone_part).item()
                        cosine_sim = F.cosine_similarity(
                            original_state.flatten(), 
                            eomt_backbone_part.flatten(), 
                            dim=0
                        ).item()
                        
                        status = "✅" if match else "❌"
                        if i >= 20:  # 只打印最后几层
                            print(f"    Layer {i}: {status} MSE={mse:.2e}, 余弦相似性={cosine_sim:.6f}")
                        
                        if not match:
                            print(f"      ⚠️ Layer {i} backbone部分有差异！")
                            return i
        
        print(f"\n  ✅ 所有层处理正确！")
        return None

def main():
    """主函数"""
    print("🔍 完整24层对比测试（关注query tokens插入）")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    original, eomt = load_models()
    original = original.to(device).eval()
    eomt = eomt.to(device).eval()
    
    print(f"  EoMT配置: num_blocks={eomt.num_blocks}, num_q={eomt.num_q}")
    print(f"  Query插入位置: 第{len(eomt.encoder.backbone.blocks) - eomt.num_blocks}层开始")
    
    # 2. 创建归一化输入
    print("\n2. 创建归一化输入...")
    normalized_input = create_normalized_input().to(device)
    print(f"归一化输入: shape={normalized_input.shape}")
    print(f"            mean={normalized_input.mean():.6f}, std={normalized_input.std():.6f}")
    
    # 3. 完整对比
    diff_layer = compare_original_vs_eomt(original, eomt, normalized_input)
    
    if diff_layer is not None:
        print(f"\n❌ 在Layer {diff_layer}开始出现差异")
    else:
        print(f"\n✅ 所有24层处理完全正确！")
    
    # 4. 测试masked attention的影响
    print(f"\n3. 测试Masked Attention的影响...")
    raw_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        # 无masked attention
        eomt.masked_attn_enabled = False
        mask_logits_no_mask, class_logits_no_mask = eomt(raw_input)
        
        # 有masked attention
        eomt.masked_attn_enabled = True
        mask_logits_with_mask, class_logits_with_mask = eomt(raw_input)
    
    print(f"  无masked attention: {len(mask_logits_no_mask)}层输出")
    print(f"  有masked attention: {len(mask_logits_with_mask)}层输出")
    
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
    
    print(f"\n{'='*70}")
    print("📋 关键发现:")
    print("1. 前20层（插入query前）：原版DINOv3 vs EoMT应该完全一致")
    print("2. 第20层（插入query）：形状改变，但backbone部分应该一致")
    print("3. 后4层（有query）：backbone部分应该继续一致")
    print("4. Masked attention：在此基础上的进一步优化")
    print("="*70)

if __name__ == "__main__":
    main()
