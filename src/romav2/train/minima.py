"""Fine-tuning on MINIMA-format data (MegaDepth + MegaDepth-Syn).

MINIMA keeps the synthetic modalities next to the MegaDepth images, one directory per
modality holding the same relative paths (``<root>/infrared/phoenix/.../x.jpg``). As in
MINIMA's RoMa training, every pair keeps the MegaDepth geometry (depth, intrinsics,
poses) and one randomly chosen side is replaced by a randomly chosen modality, which
yields cross-modal pairs when the modality is not ``visible``.

Run like ``python -m romav2.train``, with the extra ``--modality-list`` flag::

    torchrun --nproc_per_node 4 -m romav2.train.minima --stage matcher \\
        --megadepth data/megadepth --modality-list visible infrared depth normal \\
        --out runs/minima_romav2
"""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Sequence
from dataclasses import MISSING, fields
from pathlib import Path

from PIL import Image

from romav2.train.data import JsonPairDataset, MegaDepthDataset, PairDataset
from romav2.train.train import TrainCfg, train

logger = logging.getLogger(__name__)

# File extension the data engine of MINIMA writes per modality ('visible' = MegaDepth).
MODALITY_SUFFIX = {
    "visible": None,
    "infrared": ".jpg",
    "depth": ".jpg",
    "normal": ".jpg",
    "sketch": ".jpg",
    "paint": ".jpg",
    "event": ".png",
}


class MinimaMegaDepthDataset(MegaDepthDataset):
    """MegaDepth pairs in which one image is swapped for its MINIMA modality version.

    ``modality_root`` defaults to ``data_root``, i.e. the modality directories are
    expected at ``<data_root>/<modality>/`` (the layout the MINIMA README symlinks to).
    """

    def __init__(
        self,
        data_root: str | Path = "data/megadepth",
        modality_list: Sequence[str] = ("visible",),
        modality_root: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(data_root, **kwargs)
        unknown = set(modality_list) - set(MODALITY_SUFFIX)
        if unknown:
            raise ValueError(f"Unknown modalities {sorted(unknown)}, known: {sorted(MODALITY_SUFFIX)}")
        self.modalities = list(modality_list)
        self.modality_root = Path(modality_root) if modality_root is not None else self.data_root
        logger.info(f"MINIMA modalities: {self.modalities} (root {self.modality_root})")

    def load_pair(self, idx: int) -> dict:
        pair = super().load_pair(idx)
        modality = random.choice(self.modalities)
        if modality == "visible":
            return pair
        scene_idx, i_A, i_B = self.pairs[idx]
        image_paths = self.scenes[scene_idx]["image_paths"]
        key, i = random.choice((("img_A", i_A), ("img_B", i_B)))
        path = (self.modality_root / modality / image_paths[i]).with_suffix(MODALITY_SUFFIX[modality])
        if not path.exists():  # MegaDepth-Syn does not cover every MegaDepth image
            return pair
        pair[key] = Image.open(path)
        return pair


def parse_args() -> tuple[TrainCfg, list[str], str | None]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(TrainCfg):
        name = "--" + f.name.replace("_", "-")
        default = f.default if f.default is not MISSING else f.default_factory()
        if f.type == "bool":
            parser.add_argument(name, action="store_true", default=default)
        elif f.type == "list[str]":
            parser.add_argument(name, nargs="*", default=default)
        else:
            base = str if f.default is None else type(default)
            parser.add_argument(name, type=base, default=default)
    parser.add_argument("--modality-list", nargs="+", default=["visible"], help=f"any of {sorted(MODALITY_SUFFIX)}")
    parser.add_argument("--modality-root", default=None, help="parent of the modality directories (default: --megadepth)")
    args = vars(parser.parse_args())
    modality_list = args.pop("modality_list")
    modality_root = args.pop("modality_root")
    return TrainCfg(**args), modality_list, modality_root


def main():
    cfg, modality_list, modality_root = parse_args()
    if cfg.megadepth is None:
        raise SystemExit("--megadepth <MINIMA megadepth root> is required")
    kwargs = {"modality_list": modality_list, "modality_root": modality_root}
    datasets: list[PairDataset] = [MinimaMegaDepthDataset(cfg.megadepth, split="train", **kwargs)]
    weights = [1.0]
    for index in cfg.pairs:
        d = JsonPairDataset(index)
        datasets.append(d)
        weights.append(d.weight)
    if cfg.val == "megadepth":
        val_dataset = MinimaMegaDepthDataset(cfg.megadepth, split="test", augment=False, **kwargs)
    elif cfg.val is not None:
        val_dataset = JsonPairDataset(cfg.val, augment=False)
    else:
        val_dataset = None
    train(cfg, datasets=datasets, weights=weights, val_dataset=val_dataset)


if __name__ == "__main__":
    main()
