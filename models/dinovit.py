# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import timm
import torch
import torch.nn as nn

from models.dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l
from models.dinov2.models.vision_transformer import DinoVisionTransformer


class DinoViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=14,
        backbone_name="dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l()
        self.backbone: DinoVisionTransformer = self.model.visual_model.backbone.model

        self.path_grid_size = (
            img_size[0] // patch_size,
            img_size[1] // patch_size,
        )

        pixel_mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor((0.229, 0.224, 0.225)).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
