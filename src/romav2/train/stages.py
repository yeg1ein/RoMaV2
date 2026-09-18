"""The two training stages of RoMa v2 (Sec. 3.1): first the coarse matcher, then the
refiners on top of the frozen matcher. Each stage is an ``nn.Module`` whose forward
returns the loss terms, so it can be wrapped in ``DistributedDataParallel`` as is.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from romav2.geometry import bhwc_interpolate
from romav2.romav2 import RoMaV2
from romav2.train.data import ground_truth
from romav2.train.losses import matcher_loss, refiner_loss


def _sum_directions(losses_AB: dict, losses_BA: dict | None) -> dict[str, torch.Tensor]:
    if losses_BA is None:
        return losses_AB
    return {k: losses_AB[k] + losses_BA[k] for k in losses_AB}


class MatcherStage(nn.Module):
    """Stage 1: trains ``model.matcher`` on frozen DINOv3 features (Sec. 3.2)."""

    strides = (4, 16)  # warp / overlap head at stride 4, L_NLL on the patch grid

    def __init__(self, model: RoMaV2, bidirectional: bool = False):
        super().__init__()
        self.model = model
        self.bidirectional = bidirectional

    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"matcher": self.model.matcher}

    def _match(self, img_A: torch.Tensor, img_B: torch.Tensor, bidirectional: bool):
        f_A, f_B = self.model.f(img_A), self.model.f(img_B)
        return self.model.matcher(f_A, f_B, img_A=img_A, img_B=img_B, bidirectional=bidirectional)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self._match(batch["img_A"], batch["img_B"], self.bidirectional)
        gt = ground_truth(batch["warp_A_to_B"], batch["mask_A_to_B"], self.strides)
        losses = matcher_loss(
            warp=out["warp_AB"], confidence=out["confidence_AB"], attn_logits=out["attn_AB_logits"], gt=gt
        )
        losses_BA = None
        if self.bidirectional:
            gt = ground_truth(batch["warp_B_to_A"], batch["mask_B_to_A"], self.strides)
            losses_BA = matcher_loss(
                warp=out["warp_BA"], confidence=out["confidence_BA"], attn_logits=out["attn_BA_logits"], gt=gt
            )
        return _sum_directions(losses, losses_BA)

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Coarse warp (B, H, W, 2) and overlap (B, H, W), upsampled to the image size."""
        H, W = batch["img_A"].shape[-2:]
        out = self._match(batch["img_A"], batch["img_B"], bidirectional=False)
        warp = bhwc_interpolate(out["warp_AB"], (H, W), mode="bilinear", align_corners=False)
        overlap = bhwc_interpolate(out["confidence_AB"], (H, W), mode="bilinear", align_corners=False)
        return warp, overlap[..., 0].sigmoid()


class RefinerStage(nn.Module):
    """Stage 2: trains ``model.refiners`` and ``model.refiner_features`` (VGG19) on the
    coarse warps of the frozen matcher (Sec. 3.3)."""

    strides = (4, 2, 1)

    def __init__(self, model: RoMaV2, bidirectional: bool = False):
        super().__init__()
        self.model = model
        self.bidirectional = bidirectional

    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"refiners": self.model.refiners, "refiner_features": self.model.refiner_features}

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.matcher.eval()  # frozen, always in inference mode
        return self

    @torch.no_grad()
    def _coarse(self, img_A: torch.Tensor, img_B: torch.Tensor, bidirectional: bool):
        f_A, f_B = self.model.f(img_A), self.model.f(img_B)
        return self.model.matcher(f_A, f_B, img_A=img_A, img_B=img_B, bidirectional=bidirectional)

    def _refine(
        self,
        f_A: dict[int, torch.Tensor],
        f_B: dict[int, torch.Tensor],
        warp: torch.Tensor,
        confidence: torch.Tensor,
        image_size: tuple[int, int],
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Run the stride-4, 2, 1 refiners (same as ``RoMaV2.forward`` at a single resolution)."""
        H, W = image_size
        scale_factor = torch.tensor(
            (W / self.model.anchor_width, H / self.model.anchor_height), device=warp.device
        )
        outputs = {}
        for stride_str, refiner in self.model.refiners.items():
            stride = int(stride_str)
            size = (H // stride, W // stride)
            warp = bhwc_interpolate(warp.detach(), size, mode="bilinear", align_corners=False)
            confidence = bhwc_interpolate(confidence.detach(), size, mode="bilinear", align_corners=False)
            out = refiner(
                f_A=f_A[stride],
                f_B=f_B[stride],
                prev_warp=warp,
                prev_confidence=confidence,
                scale_factor=scale_factor,
            )
            warp, confidence = out["warp"], out["confidence"]
            outputs[stride] = out
        return outputs

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img_A, img_B = batch["img_A"], batch["img_B"]
        image_size = tuple(img_A.shape[-2:])
        coarse = self._coarse(img_A, img_B, self.bidirectional)
        f_A, f_B = self.model.refiner_features(img_A), self.model.refiner_features(img_B)

        outputs = self._refine(f_A, f_B, coarse["warp_AB"], coarse["confidence_AB"], image_size)
        gt = ground_truth(batch["warp_A_to_B"], batch["mask_A_to_B"], self.strides)
        losses = refiner_loss(outputs=outputs, gt=gt, image_size=image_size)
        losses_BA = None
        if self.bidirectional:
            outputs = self._refine(f_B, f_A, coarse["warp_BA"], coarse["confidence_BA"], image_size)
            gt = ground_truth(batch["warp_B_to_A"], batch["mask_B_to_A"], self.strides)
            losses_BA = refiner_loss(outputs=outputs, gt=gt, image_size=image_size)
        return _sum_directions(losses, losses_BA)

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Refined warp (B, H, W, 2) and overlap (B, H, W) at the image size."""
        img_A, img_B = batch["img_A"], batch["img_B"]
        coarse = self._coarse(img_A, img_B, bidirectional=False)
        f_A, f_B = self.model.refiner_features(img_A), self.model.refiner_features(img_B)
        out = self._refine(f_A, f_B, coarse["warp_AB"], coarse["confidence_AB"], tuple(img_A.shape[-2:]))[1]
        return out["warp"], out["confidence"][..., 0].sigmoid()
