"""Training data for RoMa v2 (Sec. 3.4, 3.5 and Appendix B, C of the paper).

Every dataset yields a dict with

    img_A, img_B           (3, H, W)  float in [0, 1]
    warp_A_to_B, warp_B_to_A  (H, W, 2)  normalized coordinates in [-1, 1]
    mask_A_to_B, mask_B_to_A  (H, W, 1)  binary covisibility

at a fixed working size (640 x 640 by default; the image is stretched, aspect
ratio is not preserved). Ground truth is derived either from depth + poses
(depth consistency, Eq. 17) or from warps / optical flow (cycle consistency,
Eq. 18). Because warps are normalized, they can be resized freely, which is how
the matcher is trained on several resolutions (``resize_batch``).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from romav2.geometry import (
    bhwc_interpolate,
    get_pixel_grid,
    overlap_from_warps,
    to_normalized,
    warp_and_overlap_from_depth,
)

logger = logging.getLogger(__name__)

# Sec. 3.5: resolutions (H, W) the matcher is trained on. Refiners use 640 x 640 only.
MATCHER_RESOLUTIONS = [
    (512, 512),
    (592, 448),
    (624, 416),
    (688, 384),
    (448, 592),
    (416, 624),
    (384, 688),
]
REFINER_RESOLUTION = (640, 640)

REL_DEPTH_ERROR_THRESHOLD = 0.05  # Eq. 17
CYCLE_ERROR_THRESHOLD = 5e-3  # Eq. 18


# ----------------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------------


def warps_from_depth(
    depth_A: torch.Tensor,
    depth_B: torch.Tensor,
    K_A: torch.Tensor,
    K_B: torch.Tensor,
    T_AB: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bidirectional GT warps and covisibility masks from depth (Eq. 17).

    ``depth``: (H, W), 0 or NaN marks invalid depth. ``K``: (3, 3) at the depth
    resolution. ``T_AB``: (4, 4) mapping camera-A coordinates to camera B.
    Returns warp_A_to_B (H, W, 2), mask_A_to_B (H, W, 1), warp_B_to_A, mask_B_to_A.
    """
    kwargs = {"rel_depth_error_threshold": REL_DEPTH_ERROR_THRESHOLD, "mode": "covis"}
    d_A, d_B = depth_A[None, ..., None], depth_B[None, ..., None]
    warp_AB, mask_AB = warp_and_overlap_from_depth(
        depth_A=d_A, depth_B=d_B, K_A=K_A[None], K_B=K_B[None], T_AB=T_AB[None], **kwargs
    )
    warp_BA, mask_BA = warp_and_overlap_from_depth(
        depth_A=d_B, depth_B=d_A, K_A=K_B[None], K_B=K_A[None], T_AB=T_AB.inverse()[None], **kwargs
    )
    return warp_AB[0], mask_AB[0], warp_BA[0], mask_BA[0]


def flow_to_warp(flow: torch.Tensor) -> torch.Tensor:
    """Pixel optical flow (H, W, 2) -> normalized warp (H, W, 2)."""
    H, W, _ = flow.shape
    grid = get_pixel_grid(1, H=H, W=W, overload_device=flow.device)[0]
    return to_normalized(grid + flow, H=H, W=W)


def ground_truth_at(
    warp: torch.Tensor, mask: torch.Tensor, size: tuple[int, int]
) -> dict[str, torch.Tensor]:
    """Resample full-resolution GT (B, H, W, 2) / (B, H, W, 1) to a prediction resolution.

    ``covisible`` (strict, all neighbours covisible) is used for regression losses,
    ``overlap`` (binary) is the target of the overlap BCE.
    """
    warp_s = bhwc_interpolate(warp, size, mode="bilinear", align_corners=False)
    mask_s = bhwc_interpolate(mask, size, mode="bilinear", align_corners=False)[..., 0]
    return {"warp": warp_s, "covisible": mask_s > 0.99, "overlap": (mask_s > 0.5).float()}


def ground_truth(
    warp: torch.Tensor, mask: torch.Tensor, strides: tuple[int, ...]
) -> dict[str, dict[str, torch.Tensor]]:
    """Ground truth for every prediction stride, keyed by ``str(stride)``."""
    _, H, W, _ = warp.shape
    return {str(s): ground_truth_at(warp, mask, (H // s, W // s)) for s in strides}


def resize_batch(batch: dict[str, torch.Tensor], size: tuple[int, int]) -> dict[str, torch.Tensor]:
    """Resize a (device) batch to another working resolution (Sec. 3.5, matcher training)."""
    out = dict(batch)
    for k in ("img_A", "img_B"):
        out[k] = F.interpolate(
            batch[k], size=size, mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)
    for k in ("warp_A_to_B", "warp_B_to_A", "mask_A_to_B", "mask_B_to_A"):
        out[k] = bhwc_interpolate(batch[k], size, mode="bilinear", align_corners=False)
    return out


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in (".h5", ".hdf5"):
        import h5py  # optional dependency (romav2[train])

        with h5py.File(path, "r") as f:
            return np.array(f["depth"])
    raise ValueError(f"Unsupported array file: {path}")


class PairDataset(Dataset):
    """Base class: subclasses implement ``load_pair(idx)`` returning a dict with

    ``img_A``, ``img_B`` (PIL images or uint8 HxWx3 arrays) and either
      * depth style: ``depth_A``, ``depth_B`` (H, W), ``K_A``, ``K_B`` (3, 3) at the image
        resolution, ``T_AB`` (4, 4) camera A -> camera B, or
      * warp style: ``warp_A_to_B``, ``warp_B_to_A`` (H, W, 2) normalized coordinates.

    The base class resizes to ``size``, applies the augmentations of Appendix C
    (random translation for depth datasets when ``shake_t > 0``, colour jitter,
    grayscale, horizontal flip) and computes the GT warps and covisibility.
    """

    def __init__(
        self,
        size: tuple[int, int] = REFINER_RESOLUTION,
        shake_t: int = 0,
        augment: bool = True,
    ):
        self.size = size
        self.shake_t = shake_t if augment else 0
        self.flip_prob = 0.5 if augment else 0.0
        self.photometric = (
            v2.Compose(
                [
                    v2.ColorJitter(brightness=(1 / 1.5, 1.5), hue=15 / 360),
                    v2.RandomGrayscale(p=0.1),
                ]
            )
            if augment
            else None
        )

    def load_pair(self, idx: int) -> dict:
        raise NotImplementedError

    def sample_weights(self) -> torch.Tensor:
        """Relative sampling probability of each pair (uniform by default)."""
        return torch.ones(len(self))

    # -- helpers --------------------------------------------------------------

    def _load_image(self, img) -> torch.Tensor:
        img = torch.from_numpy(np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img))
        img = img.permute(2, 0, 1)  # (3, H, W) uint8
        img = TF.resize(img, list(self.size), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        return img.float() / 255.0

    def _load_depth(self, depth, K) -> tuple[torch.Tensor, torch.Tensor]:
        depth = torch.as_tensor(np.asarray(depth), dtype=torch.float32)
        H0, W0 = depth.shape
        depth = TF.resize(depth[None], list(self.size), interpolation=TF.InterpolationMode.NEAREST_EXACT)[0]
        H, W = self.size
        scale = torch.tensor([[W / W0, 0, 0], [0, H / H0, 0], [0, 0, 1.0]])
        K = scale @ torch.as_tensor(np.asarray(K), dtype=torch.float32)
        return depth, K

    def _depth_gt(self, pair: dict, img_A: torch.Tensor, img_B: torch.Tensor):
        depth_A, K_A = self._load_depth(pair["depth_A"], pair["K_A"])
        depth_B, K_B = self._load_depth(pair["depth_B"], pair["K_B"])
        T_AB = torch.as_tensor(np.asarray(pair["T_AB"]), dtype=torch.float32)
        if self.shake_t > 0:
            # RoMa: translate images and depths jointly, shift the principal points
            t = [random.randint(-self.shake_t, self.shake_t) for _ in range(2)]
            img_A, img_B, depth_A, depth_B = [
                TF.affine(x[None], angle=0.0, translate=t, scale=1.0, shear=[0.0, 0.0])[0]
                for x in (img_A, img_B, depth_A, depth_B)
            ]
            K_A[:2, 2] += torch.tensor(t, dtype=torch.float32)
            K_B[:2, 2] += torch.tensor(t, dtype=torch.float32)
        warps = warps_from_depth(depth_A, depth_B, K_A, K_B, T_AB)
        return img_A, img_B, warps

    def _warp_gt(self, pair: dict):
        warp_AB = torch.as_tensor(np.asarray(pair["warp_A_to_B"]), dtype=torch.float32)
        warp_BA = torch.as_tensor(np.asarray(pair["warp_B_to_A"]), dtype=torch.float32)
        warp_AB = bhwc_interpolate(warp_AB[None], self.size, mode="bilinear", align_corners=False)
        warp_BA = bhwc_interpolate(warp_BA[None], self.size, mode="bilinear", align_corners=False)
        mask_AB = overlap_from_warps(warp_AB, warp_BA, error_threshold=CYCLE_ERROR_THRESHOLD)
        mask_BA = overlap_from_warps(warp_BA, warp_AB, error_threshold=CYCLE_ERROR_THRESHOLD)
        return warp_AB[0], mask_AB[0], warp_BA[0], mask_BA[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.load_pair(idx)
        img_A, img_B = self._load_image(pair["img_A"]), self._load_image(pair["img_B"])
        if "depth_A" in pair:
            img_A, img_B, warps = self._depth_gt(pair, img_A, img_B)
        else:
            warps = self._warp_gt(pair)
        warp_AB, mask_AB, warp_BA, mask_BA = warps
        if self.photometric is not None:
            img_A, img_B = self.photometric(img_A), self.photometric(img_B)
        if random.random() < self.flip_prob:
            img_A, img_B = img_A.flip(-1), img_B.flip(-1)
            warp_AB, warp_BA = _flip_warp(warp_AB), _flip_warp(warp_BA)
            mask_AB, mask_BA = mask_AB.flip(1), mask_BA.flip(1)
        return {
            "img_A": img_A,
            "img_B": img_B,
            "warp_A_to_B": warp_AB,
            "mask_A_to_B": mask_AB,
            "warp_B_to_A": warp_BA,
            "mask_B_to_A": mask_BA,
        }


def _flip_warp(warp: torch.Tensor) -> torch.Tensor:
    """Warp of horizontally flipped images: mirror the grid and negate x."""
    warp = warp.flip(1).clone()
    warp[..., 0] = -warp[..., 0]
    return warp


class MegaDepthDataset(PairDataset):
    """MegaDepth as in RoMa / RoMa v2 (Appendix B).

    Expects the DKM/RoMa layout under ``data_root``: ``prep_scene_info/<scene>.npy``
    (dicts with image_paths, depth_paths, intrinsics, poses, pairs, overlaps) and the
    image / depth files those reference. Per scene, up to ``max_pairs`` pairs with
    overlap > 0.01 and up to ``max_pairs`` with overlap > 0.35 are used; a pair is
    sampled with probability proportional to (#pairs in its scene)^-0.75.
    """

    TEST_SCENES = {"0015.npy", "0022.npy"}  # Mega-1500
    IGNORED_SCENES = {  # as in RoMa: LoFTR-ignored scenes and IMC21 test scenes
        "0121.npy", "0133.npy", "0168.npy", "0178.npy", "0229.npy", "0349.npy", "0412.npy",
        "0430.npy", "0443.npy", "1001.npy", "5014.npy", "5015.npy", "5016.npy",
        "0008.npy", "0019.npy", "0021.npy", "0024.npy", "0025.npy", "0032.npy", "0063.npy", "1589.npy",
    }  # fmt: skip

    def __init__(
        self,
        data_root: str | Path = "data/megadepth",
        split: str = "train",
        max_pairs: int = 100_000,
        shake_t: int = 32,
        seed: int = 0,
        scenes: list[str] | None = None,  # explicit scene files (e.g. ["0015.npy"]) instead of a split
        **kwargs,
    ):
        super().__init__(shake_t=shake_t, **kwargs)
        self.data_root = Path(data_root)
        names = sorted(p.name for p in (self.data_root / "prep_scene_info").glob("*.npy"))
        if scenes is not None:
            names = [n for n in names if n in set(scenes)]
        elif split == "train":
            names = [n for n in names if n not in self.TEST_SCENES | self.IGNORED_SCENES]
        elif split == "test":
            names = [n for n in names if n in self.TEST_SCENES]
        else:
            raise ValueError(f"Unknown split {split}")
        rng = np.random.default_rng(seed)
        self.scenes, pairs, self.pairs_per_scene = [], [], []
        for name in names:
            info = np.load(self.data_root / "prep_scene_info" / name, allow_pickle=True).item()
            overlaps = np.asarray(info["overlaps"])
            selected = []
            for min_overlap in (0.01, 0.35):
                inds = np.nonzero(overlaps > min_overlap)[0]
                if len(inds) > max_pairs:
                    inds = rng.choice(inds, max_pairs, replace=False)
                selected.append(inds)
            selected = np.concatenate(selected)
            scene_pairs = np.asarray(info["pairs"])[selected]
            pairs.append(np.c_[np.full(len(selected), len(self.scenes)), scene_pairs])
            self.scenes.append(info)
            self.pairs_per_scene.append(len(selected))
        self.pairs = np.concatenate(pairs).astype(np.int64)
        logger.info(f"MegaDepth {split}: {len(self.scenes)} scenes, {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def sample_weights(self) -> torch.Tensor:
        return torch.cat([torch.full((n,), n**-0.75) for n in self.pairs_per_scene])

    def load_pair(self, idx: int) -> dict:
        scene_idx, i1, i2 = self.pairs[idx]
        s = self.scenes[scene_idx]
        T1, T2 = np.asarray(s["poses"][i1]), np.asarray(s["poses"][i2])  # world -> camera
        return {
            "img_A": Image.open(self.data_root / s["image_paths"][i1]),
            "img_B": Image.open(self.data_root / s["image_paths"][i2]),
            "depth_A": _load_array(self.data_root / s["depth_paths"][i1]),
            "depth_B": _load_array(self.data_root / s["depth_paths"][i2]),
            "K_A": s["intrinsics"][i1],
            "K_B": s["intrinsics"][i2],
            "T_AB": T2 @ np.linalg.inv(T1),
        }


class JsonPairDataset(PairDataset):
    """Generic image-pair dataset described by a JSON index, for the other datasets of
    Table 3 or your own data. Format::

        {"name": "hypersim", "weight": 1.0, "root": "/data/hypersim",  # root optional
         "pairs": [ {...}, ... ]}

    Each pair has ``img_A``, ``img_B`` (image paths) and one of

      * ``depth_A``, ``depth_B`` (.npy (H, W) float, 0/NaN invalid, or MegaDepth .h5),
        ``K_A``, ``K_B`` (3x3 nested lists), ``T_AB`` (4x4 nested list, camera A -> B);
      * ``warp_A_to_B``, ``warp_B_to_A`` (.npy (H, W, 2), normalized coordinates);
      * ``flow_A_to_B``, ``flow_B_to_A`` (.npy (H, W, 2), pixel optical flow).

    Paths are relative to ``root`` (default: the directory of the index file).
    Covisibility is depth consistency (Eq. 17) or cycle consistency (Eq. 18).
    """

    def __init__(self, index: str | Path, shake_t: int = 0, **kwargs):
        super().__init__(shake_t=shake_t, **kwargs)
        index = Path(index)
        spec = json.loads(index.read_text())
        self.name = spec.get("name", index.stem)
        self.weight = float(spec.get("weight", 1.0))
        self.root = Path(spec.get("root", index.parent))
        self.pairs = spec["pairs"]
        logger.info(f"{self.name}: {len(self.pairs)} pairs (weight {self.weight})")

    def __len__(self) -> int:
        return len(self.pairs)

    def load_pair(self, idx: int) -> dict:
        p = self.pairs[idx]
        pair = {"img_A": Image.open(self.root / p["img_A"]), "img_B": Image.open(self.root / p["img_B"])}
        if "depth_A" in p:
            pair |= {
                "depth_A": _load_array(self.root / p["depth_A"]),
                "depth_B": _load_array(self.root / p["depth_B"]),
                "K_A": p["K_A"],
                "K_B": p["K_B"],
                "T_AB": p["T_AB"],
            }
        elif "warp_A_to_B" in p:
            pair |= {
                "warp_A_to_B": _load_array(self.root / p["warp_A_to_B"]),
                "warp_B_to_A": _load_array(self.root / p["warp_B_to_A"]),
            }
        else:
            pair |= {
                "warp_A_to_B": flow_to_warp(torch.from_numpy(_load_array(self.root / p["flow_A_to_B"]))),
                "warp_B_to_A": flow_to_warp(torch.from_numpy(_load_array(self.root / p["flow_B_to_A"]))),
            }
        return pair


def mixture_weights(datasets: list[PairDataset], weights: list[float]) -> torch.Tensor:
    """Per-sample weights for a ``ConcatDataset`` of ``datasets`` such that dataset i is
    drawn with probability proportional to ``weights[i]`` (Table 3), and within it
    according to ``sample_weights()``."""
    out = []
    for d, w in zip(datasets, weights):
        sw = d.sample_weights().double()
        out.append(w * sw / sw.sum())
    return torch.cat(out)
