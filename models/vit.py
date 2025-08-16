# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from ast import main
from typing import Optional
import timm
import torch
import torch.nn as nn
from models.dinov3.dinov3.models.vision_transformer import DinoVisionTransformer

class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        # self.backbone = timm.create_model(
        #     backbone_name,
        #     pretrained=ckpt_path is None,
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     num_classes=0,
        # )

        # 权重文件路径
        weights_path = '/home/ipbhk/data/projects/eomt/models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        
        try:
            # 直接使用DINOv3的函数，传入正确的img_size
            from models.dinov3.dinov3.hub.backbones import _make_dinov3_vit, Weights
            
            model = _make_dinov3_vit(
                img_size=max(img_size),  # 使用最大的尺寸
                patch_size=patch_size,
                in_chans=3,
                pos_embed_rope_base=100,
                pos_embed_rope_normalize_coords="separate",
                pos_embed_rope_rescale_coords=2,
                pos_embed_rope_dtype="fp32",
                embed_dim=1024,
                depth=24,
                num_heads=16,
                ffn_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.0,
                layerscale_init=1.0e-05,
                norm_layer="layernormbf16",
                ffn_layer="mlp",
                ffn_bias=True,
                proj_bias=True,
                n_storage_tokens=4,
                mask_k_bias=True,
                untie_global_and_local_cls_norm=False,
                pretrained=True,
                weights=weights_path,
                compact_arch_name="vitl",
                check_hash=False,
            )
            print(f"✓ 模型加载成功! 输入尺寸: {img_size}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e
        
        self.backbone: DinoVisionTransformer = model
        
        # 保存图像尺寸信息供其他模块使用
        self.img_size = img_size

        # DINOv3没有default_cfg，手动设置ImageNet标准化参数
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 权重文件路径
    weights_path = '/home/ipbhk/data/projects/eomt/models/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    
    try:
        # 使用torch.hub加载
        repo_dir = '/home/ipbhk/data/projects/eomt/models/dinov3'
        model = torch.hub.load(
            repo_dir, 
            'dinov3_vitl16', 
            source='local', 
            weights=weights_path
        )
        model = model.to(device).eval()
        print("✓ 模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
    

    print(model)