"""Training losses of RoMa v2 (Sec. 3.2 and 3.3 of the paper).

Conventions
-----------
* Warps are in normalized image coordinates ([-1, 1] x [-1, 1]), as everywhere
  else in the code base. The residual r = W_pred - W_gt is therefore also
  normalized, which is what the generalized Charbonnier loss expects
  (c = 1e-3 is in normalized units, i is the stride).
* The precision matrix is in *pixel* units of the input image (see
  ``ConvRefiner`` and ``RoMaV2.prec_map_coordinates``), so the precision NLL
  converts the residual to pixels first.
* Ground-truth overlaps p_GT are binary. ``covisible`` is the strict mask used
  for the regression-type losses, ``overlap`` is the target of the BCE loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from romav2.geometry import to_pixel


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over ``mask`` (bool). Zero (with grad) if the mask is empty.
    Uses ``where`` rather than ``x * mask`` so that non-finite masked-out entries cannot
    leak NaN into the result."""
    x = torch.where(mask, x, torch.zeros_like(x))
    return x.sum() / mask.sum().clamp_min(1)


def warp_loss(
    warp: torch.Tensor,
    gt_warp: torch.Tensor,
    covisible: torch.Tensor,
    stride: int,
    alpha: float = 0.5,
    c: float = 1e-3,
) -> torch.Tensor:
    """Generalized Charbonnier loss L_warp = (ic)^a * (|r|^2 / (ic)^2 + 1)^(a/2).

    Mean over covisible pixels. ``warp``, ``gt_warp``: (B, H, W, 2); ``covisible``: (B, H, W).
    """
    ic = c * stride
    r = (warp - gt_warp).norm(dim=-1)
    loss = ic**alpha * ((r / ic) ** 2 + 1) ** (alpha / 2)
    return masked_mean(loss, covisible)


def overlap_loss(overlap_logit: torch.Tensor, gt_overlap: torch.Tensor) -> torch.Tensor:
    """Pixel-wise binary cross-entropy L_overlap. Both (B, H, W)."""
    return F.binary_cross_entropy_with_logits(overlap_logit, gt_overlap.float())


def matching_nll_loss(
    attn_logits: torch.Tensor,
    gt_warp: torch.Tensor,
    covisible: torch.Tensor,
) -> torch.Tensor:
    """L_NLL (Eq. 1): -log softmax(S_m)[n*], n* being the patch in B closest to the GT warp.

    ``attn_logits``: (B, H_A, W_A, H_B, W_B) similarity logits (cossim / tau),
    ``gt_warp``: (B, H_A, W_A, 2) GT warp at the patch centers of A,
    ``covisible``: (B, H_A, W_A). Mean over covisible patches.
    """
    B, H_A, W_A, H_B, W_B = attn_logits.shape
    # pixel-center coordinates on the patch grid of B: floor gives the containing patch
    x = to_pixel(gt_warp, H=H_B, W=W_B)
    ix = x[..., 0].floor().long().clamp(0, W_B - 1)
    iy = x[..., 1].floor().long().clamp(0, H_B - 1)
    target = (iy * W_B + ix).reshape(-1, 1)
    log_p = torch.log_softmax(attn_logits.reshape(B * H_A * W_A, H_B * W_B), dim=-1)
    nll = -log_p.gather(1, target).reshape(B, H_A, W_A)
    return masked_mean(nll, covisible)


def precision_loss(
    prec_params: torch.Tensor,
    residual_px: torch.Tensor,
    covisible: torch.Tensor,
    max_residual_px: float = 8.0,
) -> torch.Tensor:
    """L_prec (Eq. 4): Gaussian NLL of the (detached) residual under N(0, Sigma).

    ``prec_params``: (B, H, W, 3) entries (p00, p10, p11) of Sigma^-1 = L L^T (positive
    definite by construction, see ``ConvRefiner``), ``residual_px``: (B, H, W, 2) in pixels.
    Only supervised where covisible and |r| < 8 px; the residual is detached.
    """
    r = residual_px.detach()
    p00, p10, p11 = prec_params.unbind(dim=-1)
    rx, ry = r.unbind(dim=-1)
    quad = p00 * rx * rx + 2 * p10 * rx * ry + p11 * ry * ry
    log_det = torch.log((p00 * p11 - p10 * p10).clamp_min(1e-12))
    nll = 0.5 * quad - 0.5 * log_det + math.log(2 * math.pi)
    valid = covisible & (r.norm(dim=-1) < max_residual_px)
    return masked_mean(nll, valid)


def matcher_loss(
    *,
    warp: torch.Tensor,
    confidence: torch.Tensor,
    attn_logits: torch.Tensor,
    gt: dict[str, dict[str, torch.Tensor]],
    lambda_overlap: float = 0.1,
) -> dict[str, torch.Tensor]:
    """L_matcher = L_NLL + L_warp(stride 4) + lambda * L_overlap (Eq. 2).

    ``gt`` holds the ground truth at the two matcher resolutions (see ``ground_truth_at``):
    ``gt["4"]`` for the stride-4 warp/overlap head and ``gt["16"]`` on the patch grid for L_NLL.
    """
    gt4, gt16 = gt["4"], gt["16"]
    return {
        "nll": matching_nll_loss(attn_logits, gt16["warp"], gt16["covisible"]),
        "warp": warp_loss(warp, gt4["warp"], gt4["covisible"], stride=4),
        "overlap": lambda_overlap * overlap_loss(confidence[..., 0], gt4["overlap"]),
    }


def refiner_loss(
    *,
    outputs: dict[int, dict[str, torch.Tensor]],
    gt: dict[str, dict[str, torch.Tensor]],
    image_size: tuple[int, int],
    lambda_overlap: float = 1e-2,
    lambda_precision: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """L_refiners (Eq. 5), summed over the strides {4, 2, 1}.

    ``outputs[stride]`` = {"warp": (B, H/s, W/s, 2), "confidence": (B, H/s, W/s, 4)}, where
    confidence = (overlap logit, p00, p10, p11) with the precision accumulated over strides.
    """
    H, W = image_size
    device = next(iter(outputs.values()))["warp"].device
    to_px = torch.tensor((W / 2, H / 2), device=device)
    losses = {}
    for stride, out in outputs.items():
        g = gt[str(stride)]
        warp, confidence = out["warp"], out["confidence"]
        residual_px = (warp - g["warp"]) * to_px
        losses[f"warp_{stride}"] = warp_loss(warp, g["warp"], g["covisible"], stride=stride)
        losses[f"overlap_{stride}"] = lambda_overlap * overlap_loss(
            confidence[..., 0], g["overlap"]
        )
        losses[f"precision_{stride}"] = lambda_precision * precision_loss(
            confidence[..., 1:4], residual_px, g["covisible"]
        )
    return losses
