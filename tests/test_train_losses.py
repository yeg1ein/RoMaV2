"""Analytical unit tests for the training losses and ground-truth utilities (CPU only)."""

import math

import torch

from romav2.geometry import get_normalized_grid, get_pixel_grid, to_pixel
from romav2.train.data import (
    _flip_warp,
    flow_to_warp,
    ground_truth_at,
    resize_batch,
    warps_from_depth,
)
from romav2.train.losses import (
    masked_mean,
    matching_nll_loss,
    overlap_loss,
    precision_loss,
    warp_loss,
)

CPU = torch.device("cpu")


def test_warp_loss_matches_formula():
    torch.manual_seed(0)
    warp, gt = torch.randn(2, 5, 6, 2), torch.randn(2, 5, 6, 2)
    covisible = torch.rand(2, 5, 6) > 0.3
    r = (warp - gt).norm(dim=-1)
    ic = 1e-3 * 4
    expected = (ic**0.5 * ((r / ic) ** 2 + 1) ** 0.25)[covisible].mean()
    assert torch.allclose(warp_loss(warp, gt, covisible, stride=4), expected)
    # zero residual -> (ic)^alpha, i.e. the minimum of the loss
    assert torch.allclose(warp_loss(gt, gt, covisible, stride=4), torch.tensor(ic**0.5))
    # empty mask -> 0 instead of NaN
    assert warp_loss(warp, gt, torch.zeros_like(covisible), stride=4) == 0


def test_masked_mean_and_overlap_loss():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert masked_mean(x, torch.tensor([True, False, True, False])) == 2.0
    logits = torch.tensor([[[10.0, -10.0]]])
    assert overlap_loss(logits, torch.tensor([[[1.0, 0.0]]])) < 1e-3
    assert overlap_loss(logits, torch.tensor([[[0.0, 1.0]]])) > 5


def test_matching_nll_targets_containing_patch():
    """The target of patch m must be the patch of B containing its GT warp; a peaked
    similarity on exactly that patch gives ~0 loss, on any other patch a large loss."""
    B, H, W = 1, 4, 6
    grid = get_normalized_grid(B, H, W, overload_device=CPU)
    # GT warp: shift by exactly one patch to the right, wrapping the last column out of frame
    gt_warp = grid + torch.tensor([2.0 / W, 0.0])
    covisible = gt_warp[..., 0] < 1
    target = torch.arange(H * W).reshape(H, W).roll(-1, dims=1)  # index of the patch to the right
    logits = torch.full((B, H, W, H, W), -50.0)
    logits.reshape(B * H * W, H * W)[torch.arange(H * W), target.reshape(-1)] = 50.0
    assert matching_nll_loss(logits, gt_warp, covisible) < 1e-4
    wrong = logits.roll(1, dims=-1)  # peaks moved to the neighbouring patch
    assert matching_nll_loss(wrong, gt_warp, covisible) > 50


def test_precision_loss_is_minimised_by_the_true_covariance():
    """For residuals ~ N(0, Sigma), the NLL is lowest at Sigma^-1 = true precision and the
    mean Mahalanobis distance r^T Sigma^-1 r is ~2 (chi-square with 2 dof)."""
    torch.manual_seed(0)
    L = torch.tensor([[2.0, 0.0], [0.5, 1.0]])
    cov = L @ L.T
    r = torch.distributions.MultivariateNormal(torch.zeros(2), cov).sample((1, 200, 200))
    covisible = torch.ones(1, 200, 200, dtype=torch.bool)
    P = cov.inverse()
    true = torch.tensor([P[0, 0], P[1, 0], P[1, 1]]).expand(1, 200, 200, 3)
    nll_true = precision_loss(true, r, covisible, max_residual_px=1e9)
    # analytic optimum: 0.5 * 2 + 0.5 * log det(Sigma) + log(2 pi)
    assert abs(nll_true.item() - (1 + 0.5 * torch.logdet(cov).item() + math.log(2 * math.pi))) < 0.02
    for scale in (0.5, 2.0):
        assert precision_loss(true * scale, r, covisible, max_residual_px=1e9) > nll_true
    mahalanobis = (r @ P * r).sum(-1).mean()
    assert abs(mahalanobis.item() - 2.0) < 0.05
    # residuals farther than max_residual_px are ignored
    far = r.clone()
    far[0, 0, 0] = 1e6
    assert torch.isfinite(precision_loss(true, far, covisible, max_residual_px=8.0))


def test_ground_truth_from_depth_fronto_parallel_plane():
    """Plane at depth 1, camera B shifted so that everything moves by 32 px."""
    H, W = 64, 128
    depth = torch.ones(H, W)
    K = torch.tensor([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1.0]])
    T_AB = torch.eye(4)
    T_AB[0, 3] = 32 / W
    warp_AB, mask_AB, warp_BA, mask_BA = warps_from_depth(depth, depth, K, K, T_AB)
    grid = get_normalized_grid(1, H, W, overload_device=CPU)[0]
    assert torch.allclose(warp_AB, grid + torch.tensor([2 * 32 / W, 0.0]), atol=1e-5)
    assert torch.allclose(warp_BA, grid - torch.tensor([2 * 32 / W, 0.0]), atol=1e-5)
    assert mask_AB[:, : W - 33].all() and not mask_AB[:, W - 31 :].any()
    assert mask_BA[:, 33:].all() and not mask_BA[:, :31].any()


def test_ground_truth_is_finite_with_invalid_depth():
    """Depth 0 / NaN (invalid) must give finite warps and non-covisible masks, and a loss
    with finite gradient (inf * 0 = NaN would otherwise poison masked losses)."""
    H, W = 32, 32
    depth = torch.ones(H, W)
    depth[:8] = 0.0
    depth[8:12] = float("nan")
    K = torch.tensor([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1.0]])
    T_AB = torch.eye(4)
    T_AB[2, 3] = -1.0  # camera B one unit forward: depth-0 points would land at z = -1
    warp_AB, mask_AB, warp_BA, mask_BA = warps_from_depth(depth, depth, K, K, T_AB)
    for w, m in ((warp_AB, mask_AB), (warp_BA, mask_BA)):
        assert torch.isfinite(w).all() and w.abs().max() <= 2
        assert not m[:12].any()
    pred = torch.zeros(1, H, W, 2, requires_grad=True)
    loss = warp_loss(pred, warp_AB[None], mask_AB[None, ..., 0] > 0.5, stride=1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def test_flow_to_warp_and_flip():
    H, W = 8, 10
    flow = torch.zeros(H, W, 2)
    flow[..., 0] = 3.0  # 3 px to the right
    warp = flow_to_warp(flow)
    grid = get_normalized_grid(1, H, W, overload_device=CPU)[0]
    assert torch.allclose(warp, grid + torch.tensor([2 * 3 / W, 0.0]))
    # flipping both images turns a rightward shift into a leftward one on the mirrored grid
    flipped = _flip_warp(warp)
    assert torch.allclose(flipped, grid - torch.tensor([2 * 3 / W, 0.0]))
    assert torch.allclose(to_pixel(warp, H=H, W=W), get_pixel_grid(1, H=H, W=W, overload_device=CPU)[0] + flow)


def test_ground_truth_resampling_and_resize_batch():
    H, W = 32, 32
    grid = get_normalized_grid(2, H, W, overload_device=CPU)
    warp = grid * 0.5  # linear warp: bilinear resampling is exact
    mask = torch.ones(2, H, W, 1)
    mask[:, :, : W // 2] = 0  # left half occluded
    gt = ground_truth_at(warp, mask, (H // 4, W // 4))
    assert torch.allclose(gt["warp"], get_normalized_grid(2, H // 4, W // 4, overload_device=CPU) * 0.5, atol=1e-6)
    assert gt["covisible"].shape == (2, H // 4, W // 4)
    assert not gt["covisible"][:, :, : W // 8].any() and gt["covisible"][:, :, W // 8 + 1 :].all()
    batch = {"img_A": torch.rand(2, 3, H, W), "img_B": torch.rand(2, 3, H, W), "warp_A_to_B": warp, "warp_B_to_A": warp, "mask_A_to_B": mask, "mask_B_to_A": mask}
    out = resize_batch(batch, (48, 24))
    assert out["img_A"].shape == (2, 3, 48, 24) and out["warp_A_to_B"].shape == (2, 48, 24, 2)
    # normalized warps are invariant to resizing (up to bilinear clamping at the border pixels)
    expected = get_normalized_grid(2, 48, 24, overload_device=CPU) * 0.5
    assert torch.allclose(out["warp_A_to_B"][:, 1:-1, 1:-1], expected[:, 1:-1, 1:-1], atol=1e-6)
    assert (out["warp_A_to_B"] - expected).abs().max() < 0.5 * 2 / 24


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
