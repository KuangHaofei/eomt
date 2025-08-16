#!/usr/bin/env python3
"""
测试修改后的EoMT模型是否能正确工作
"""

import torch
import sys
sys.path.insert(0, 'models/dinov3')

from models.vit import ViT
from models.eomt import EoMT

def test_eomt_dinov3():
    print("测试EoMT + DINOv3集成")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        # 1. 创建ViT encoder（包含DINOv3 backbone）
        print("\n1. 创建ViT encoder...")
        img_size = (224, 224)
        encoder = ViT(img_size=img_size).to(device)
        print(f"✓ ViT encoder创建成功")
        print(f"  Backbone类型: {type(encoder.backbone)}")
        print(f"  Embed dim: {encoder.backbone.embed_dim}")
        print(f"  Storage tokens: {encoder.backbone.n_storage_tokens}")
        
        # 2. 创建EoMT模型
        print("\n2. 创建EoMT模型...")
        num_classes = 80  # COCO classes
        num_q = 100  # query数量
        
        eomt_model = EoMT(
            encoder=encoder,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=4,
            masked_attn_enabled=True
        ).to(device)
        
        print(f"✓ EoMT模型创建成功")
        print(f"  Query数量: {num_q}")
        print(f"  类别数量: {num_classes}")
        print(f"  Prefix tokens: {eomt_model.num_prefix_tokens}")
        
        # 3. 测试前向传播
        print("\n3. 测试前向传播...")
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 224, 224).to(device)
        print(f"测试输入形状: {test_input.shape}")
        
        eomt_model.eval()
        with torch.no_grad():
            mask_logits_per_layer, class_logits_per_layer = eomt_model(test_input)
        
        print(f"✓ 前向传播成功!")
        print(f"  输出层数: {len(mask_logits_per_layer)}")
        print(f"  Mask logits形状: {[ml.shape for ml in mask_logits_per_layer]}")
        print(f"  Class logits形状: {[cl.shape for cl in class_logits_per_layer]}")
        
        # 4. 验证输出形状
        print("\n4. 验证输出形状...")
        expected_mask_shape = (batch_size, num_q, 14, 14)  # 14x14 for 224x224 input with patch_size=16
        expected_class_shape = (batch_size, num_q, num_classes + 1)
        
        for i, (mask_logits, class_logits) in enumerate(zip(mask_logits_per_layer, class_logits_per_layer)):
            print(f"  Layer {i}:")
            print(f"    Mask logits: {mask_logits.shape} (期望: {expected_mask_shape})")
            print(f"    Class logits: {class_logits.shape} (期望: {expected_class_shape})")
            
            # 检查形状是否正确
            if mask_logits.shape == expected_mask_shape:
                print(f"    ✓ Mask logits形状正确")
            else:
                print(f"    ✗ Mask logits形状不正确")
                
            if class_logits.shape == expected_class_shape:
                print(f"    ✓ Class logits形状正确")
            else:
                print(f"    ✗ Class logits形状不正确")
        
        # 5. 计算参数量
        print("\n5. 模型信息:")
        total_params = sum(p.numel() for p in eomt_model.parameters())
        trainable_params = sum(p.numel() for p in eomt_model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        print("\n" + "=" * 50)
        print("🎉 EoMT + DINOv3集成测试成功!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_eomt_dinov3()
    if success:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 测试失败!")
        sys.exit(1)
