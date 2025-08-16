#!/usr/bin/env python3
"""
测试EoMT模型对不同分辨率的支持
"""

import torch
import sys
sys.path.insert(0, 'models/dinov3')

from models.vit import ViT
from models.eomt import EoMT

def test_resolution_support():
    print("测试EoMT模型对不同分辨率的支持")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试不同的分辨率
    test_resolutions = [
        (224, 224),   # 标准分辨率
        (640, 640),   # 分割任务常用分辨率
        (512, 512),   # 另一个常用分辨率
        (448, 448),   # 14的倍数
    ]
    
    results = []
    
    for img_size in test_resolutions:
        print(f"\n{'='*20} 测试分辨率 {img_size} {'='*20}")
        
        try:
            # 1. 创建ViT encoder
            print(f"1. 创建ViT encoder (分辨率: {img_size})...")
            encoder = ViT(img_size=img_size).to(device)
            print(f"✓ ViT encoder创建成功")
            
            # 2. 创建EoMT模型
            print("2. 创建EoMT模型...")
            num_classes = 80
            num_q = 100
            
            eomt_model = EoMT(
                encoder=encoder,
                num_classes=num_classes,
                num_q=num_q,
                num_blocks=4,
                masked_attn_enabled=True
            ).to(device)
            
            print(f"✓ EoMT模型创建成功")
            
            # 3. 测试前向传播
            print("3. 测试前向传播...")
            batch_size = 1  # 使用较小的batch size以节省内存
            test_input = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device)
            
            eomt_model.eval()
            with torch.no_grad():
                mask_logits_per_layer, class_logits_per_layer = eomt_model(test_input)
            
            # 4. 验证输出
            patch_size = 16
            expected_grid_h = img_size[0] // patch_size
            expected_grid_w = img_size[1] // patch_size
            expected_upscaled_h = expected_grid_h * 4  # upscale模块的放大倍数
            expected_upscaled_w = expected_grid_w * 4
            
            mask_shape = mask_logits_per_layer[0].shape
            class_shape = class_logits_per_layer[0].shape
            
            print(f"✓ 前向传播成功!")
            print(f"  输入形状: {test_input.shape}")
            print(f"  Grid size: {expected_grid_h}x{expected_grid_w}")
            print(f"  Mask logits: {mask_shape}")
            print(f"  Class logits: {class_shape}")
            print(f"  预期mask形状: ({batch_size}, {num_q}, {expected_upscaled_h}, {expected_upscaled_w})")
            
            # 记录结果
            results.append({
                'resolution': img_size,
                'success': True,
                'mask_shape': mask_shape,
                'class_shape': class_shape,
                'grid_size': (expected_grid_h, expected_grid_w),
                'error': None
            })
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results.append({
                'resolution': img_size,
                'success': False,
                'mask_shape': None,
                'class_shape': None,
                'grid_size': None,
                'error': str(e)
            })
        
        finally:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 总结结果
    print(f"\n{'='*60}")
    print("测试结果总结")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"成功率: {success_count}/{total_count}")
    
    print("\n详细结果:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['resolution']}: ", end="")
        
        if result['success']:
            print(f"Grid {result['grid_size']}, Mask {result['mask_shape']}")
        else:
            print(f"失败 - {result['error']}")
    
    # 验证分辨率缩放关系
    if success_count >= 2:
        print(f"\n{'='*30}")
        print("分辨率缩放验证:")
        print(f"{'='*30}")
        
        for i, result in enumerate(results):
            if result['success']:
                res = result['resolution']
                mask_h, mask_w = result['mask_shape'][2], result['mask_shape'][3]
                grid_h, grid_w = result['grid_size']
                
                print(f"{res}: Grid({grid_h}x{grid_w}) -> Mask({mask_h}x{mask_w})")
                print(f"  缩放比例: {mask_h/grid_h:.1f}x")
    
    return results

if __name__ == "__main__":
    results = test_resolution_support()
    
    success_count = sum(1 for r in results if r['success'])
    if success_count == len(results):
        print(f"\n🎉 所有分辨率测试通过!")
    else:
        print(f"\n⚠️ 部分测试失败，请检查错误信息")
