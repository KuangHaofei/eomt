# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.scale_block import ScaleBlock


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )
        
        # DINOv3特定属性：计算prefix tokens数量 (cls_token + storage_tokens)
        self.num_prefix_tokens = 1 + self.encoder.backbone.n_storage_tokens  # cls + storage tokens

    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        # DINOv3适配：跳过query tokens和prefix tokens (cls + storage)
        x = x[:, self.num_q + self.num_prefix_tokens :, :]
        
        # 动态计算grid_size，支持任意输入尺寸
        patch_size = self.encoder.backbone.patch_embed.patch_size[0]  # (16, 16) -> 16
        
        # 从patch tokens数量推导grid_size
        num_patch_tokens = x.shape[1]  # 去掉query和prefix tokens后的patch tokens数量
        grid_size = int(math.sqrt(num_patch_tokens))
        
        # 验证是否为完全平方数
        if grid_size * grid_size != num_patch_tokens:
            # 如果不是完全平方数，从encoder获取图像尺寸信息
            if hasattr(self.encoder, 'img_size'):
                img_size = max(self.encoder.img_size)
                grid_size = img_size // patch_size
            else:
                # 最后的fallback
                grid_size = int(math.sqrt(num_patch_tokens))
        
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, grid_size, grid_size
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(self, module: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor], rope=None):
        B, N, C = x.shape

        # 计算head_dim（DINOv3没有直接的head_dim属性）
        head_dim = C // module.num_heads
        
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # 应用RoPE位置编码（如果提供）
        if rope is not None:
            q, k = module.apply_rope(q, k, rope)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        # DINOv3没有fused_attn属性，使用标准attention计算
        attn = (q @ k.transpose(-2, -1)) * module.scale
        if mask is not None:
            # 数值稳定性改进：使用更安全的mask填充值
            min_value = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(~mask, min_value)
        
        # 检查attention矩阵是否包含全部-inf的行（数值稳定性检查）
        if mask is not None:
            # 如果某行全为False，则该行在softmax后会变为NaN
            # 检查是否存在全为False的mask行
            all_masked = (~mask).all(dim=-1, keepdim=True)
            if all_masked.any():
                # 对于全被mask的行，我们给第一个位置一个小的正值
                attn = torch.where(all_masked.expand_as(attn), 
                                 torch.full_like(attn, min_value).scatter(-1, torch.zeros_like(attn[..., :1]).long(), -1e4), 
                                 attn)
        
        attn = F.softmax(attn, dim=-1)
        attn = module.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = module.proj(x)
        x = module.proj_drop(x)

        return x

    def forward(self, x: torch.Tensor):
        # DINOv3预处理
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        # 使用DINOv3的标准token准备方法，确保与原始实现完全一致
        x, (H, W) = self.encoder.backbone.prepare_tokens_with_masks(x)

        # 位置编码（DINOv3使用RoPE，在block中处理）
        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            # 在最后几个block中添加query tokens
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            # 每个block都重新计算RoPE位置编码（与DINOv3标准流程一致）
            if self.encoder.backbone.rope_embed is not None:
                rope_or_rope_list = self.encoder.backbone.rope_embed(H=H, W=W)
            else:
                rope_or_rope_list = None

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                # 使用DINOv3的norm层
                if hasattr(self.encoder.backbone, 'norm'):
                    normed_x = self.encoder.backbone.norm(x)
                else:
                    # 如果没有全局norm，使用block的norm
                    normed_x = x
                    
                mask_logits, class_logits = self._predict(normed_x)
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                # 创建attention mask
                attn_mask = torch.ones(
                    x.shape[0],
                    x.shape[1],
                    x.shape[1],
                    dtype=torch.bool,
                    device=x.device,
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
                    : self.num_q,
                    self.num_q + self.num_prefix_tokens :,
                ] = (
                    interpolated > 0
                )
                attn_mask = self._disable_attn_mask(
                    attn_mask,
                    self.attn_mask_probs[
                        i - len(self.encoder.backbone.blocks) + self.num_blocks
                    ],
                )

                # 使用自定义的attention来处理mask，遵循DINOv3的block结构
                # 使用我们的_attn方法来处理masked attention，同时应用RoPE
                x = x + block.ls1(
                    self._attn(block.attn, block.norm1(x), attn_mask, rope_or_rope_list)
                )
                x = x + block.ls2(block.mlp(block.norm2(x)))
            else:
                # 没有mask时，直接使用DINOv3的block forward
                x = block(x, rope_or_rope_list)

        # 最终预测
        if hasattr(self.encoder.backbone, 'norm'):
            final_x = self.encoder.backbone.norm(x)
        else:
            final_x = x
            
        mask_logits, class_logits = self._predict(final_x)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
