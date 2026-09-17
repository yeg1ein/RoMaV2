"""Smoke test of the training code: builds a tiny synthetic dataset (warp-, flow- and
depth-style pairs through ``JsonPairDataset``), checks the ground truth, and runs a few
steps of both training stages, including checkpoint export and reload.

Run with ``uv run tests/test_train_smoke.py`` (needs a GPU and the released weights).
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from romav2 import RoMaV2
from romav2.geometry import get_normalized_grid, get_pixel_grid, to_pixel
from romav2.train import JsonPairDataset, TrainCfg, train

ASSET = Path(__file__).parent.parent / "assets" / "toronto_A.jpg"
H, W = 480, 640  # source resolution of the synthetic pairs
CPU = torch.device("cpu")


def make_synthetic_dataset(root: Path, num_pairs: int = 4) -> Path:
    """Image pairs related by a random similarity (warp / flow style) plus one
    fronto-parallel plane pair with camera translation (depth style)."""
    root.mkdir(parents=True, exist_ok=True)
    img = Image.open(ASSET).convert("RGB").resize((W, H))
    A = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
    grid = get_normalized_grid(1, H, W, overload_device=CPU)[0]
    pixel_grid = get_pixel_grid(1, H=H, W=W, overload_device=CPU)[0]
    pairs = []
    torch.manual_seed(0)
    for i in range(num_pairs):
        s, theta = 0.8 + 0.4 * torch.rand(()), math.radians(-20 + 40 * torch.rand(()))
        M = s * torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        t = 0.2 * torch.rand(2) - 0.1
        warp_AB = grid @ M.T + t  # x_B = M x_A + t
        warp_BA = (grid - t) @ M.inverse().T
        B = F.grid_sample(A[None], warp_BA[None], align_corners=False)[0]
        Image.fromarray((B.permute(1, 2, 0) * 255).byte().numpy()).save(root / f"B_{i}.png")
        np.save(root / f"warp_AB_{i}.npy", warp_AB.numpy())
        np.save(root / f"warp_BA_{i}.npy", warp_BA.numpy())
        if i % 2 == 0:
            pairs.append({"img_A": "A.png", "img_B": f"B_{i}.png", "warp_A_to_B": f"warp_AB_{i}.npy", "warp_B_to_A": f"warp_BA_{i}.npy"})
        else:  # same pair, expressed as pixel optical flow
            np.save(root / f"flow_AB_{i}.npy", (to_pixel(warp_AB, H=H, W=W) - pixel_grid).numpy())
            np.save(root / f"flow_BA_{i}.npy", (to_pixel(warp_BA, H=H, W=W) - pixel_grid).numpy())
            pairs.append({"img_A": "A.png", "img_B": f"B_{i}.png", "flow_A_to_B": f"flow_AB_{i}.npy", "flow_B_to_A": f"flow_BA_{i}.npy"})
    img.save(root / "A.png")
    # depth style: plane at depth 1, camera B translated so that everything shifts by 32 px
    shift = 32
    Image.fromarray(np.roll(np.asarray(img), shift, axis=1)).save(root / "B_depth.png")
    np.save(root / "depth.npy", np.ones((H, W), dtype=np.float32))
    K = [[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]]
    T_AB = np.eye(4)
    T_AB[0, 3] = shift / W
    pairs.append({"img_A": "A.png", "img_B": "B_depth.png", "depth_A": "depth.npy", "depth_B": "depth.npy", "K_A": K, "K_B": K, "T_AB": T_AB.tolist()})
    index = root / "index.json"
    index.write_text(json.dumps({"name": "synthetic", "weight": 1.0, "pairs": pairs}))
    return index


def check_ground_truth(dataset: JsonPairDataset):
    # depth pair: warp must be a 32 px shift in x at the 640 x 640 working resolution
    sample = dataset[len(dataset) - 1]
    h, w = dataset.size
    grid = get_normalized_grid(1, h, w, overload_device=CPU)[0]
    expected = grid + torch.tensor([2 * 32 / w, 0.0])
    covisible = sample["mask_A_to_B"][..., 0] > 0.5
    assert 0.8 < covisible.float().mean() < 0.97, covisible.float().mean()
    assert (sample["warp_A_to_B"] - expected)[covisible].abs().max() < 1e-4
    assert not covisible[:, -1].any()  # last columns map outside of B
    # warp / flow pairs describe the same transformation
    a, b = dataset[0], dataset[1]
    for s in (a, b):
        assert s["img_A"].shape == (3, h, w) and s["warp_A_to_B"].shape == (h, w, 2)
        assert s["mask_A_to_B"].shape == (h, w, 1) and s["mask_A_to_B"].sum() > 0


def test_train_smoke():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        index = make_synthetic_dataset(root / "data")
        dataset = JsonPairDataset(index, augment=False)
        check_ground_truth(dataset)
        common = {"steps": 4, "batch_size": 2, "workers": 0, "warmup": 1, "log_every": 1, "ckpt_every": 2, "eval_every": 2, "eval_batches": 1}
        # stage 1: matcher (fine-tuning from the released weights, bidirectional)
        train(TrainCfg(stage="matcher", out=str(root / "matcher"), pairs=[str(index)], val=str(index), bidirectional=True, **common))
        matcher_weights = root / "matcher" / "romav2_matcher.pt"
        assert matcher_weights.exists() and (root / "matcher" / "checkpoint.pt").exists()
        # stage 2: refiners, initialised from the stage-1 export, resumed once
        train(TrainCfg(stage="refiners", out=str(root / "refiners"), pairs=[str(index)], init=str(matcher_weights), **common))
        train(TrainCfg(stage="refiners", out=str(root / "refiners"), pairs=[str(index)], resume=str(root / "refiners" / "checkpoint.pt"), **{**common, "steps": 6}))
        # the export is a complete RoMaV2 state dict
        model = RoMaV2(RoMaV2.Cfg(pretrained=False, setting="turbo"))
        model.load_state_dict(torch.load(root / "refiners" / "romav2_refiners.pt"))
        preds = model.match(str(root / "data" / "A.png"), str(root / "data" / "B_0.png"))
        assert torch.isfinite(preds["warp_AB"]).all()


if __name__ == "__main__":
    test_train_smoke()
