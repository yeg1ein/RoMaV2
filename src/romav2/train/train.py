"""Training loop for RoMa v2.

Stage "matcher" (Sec. 3.2): 300k steps, batch 128, lr 4e-4, random resolution per batch.
Stage "refiners" (Sec. 3.3): 300k steps, batch 64, lr 4e-4, 640 x 640, EMA (0.999).
Both stages use AdamW. The paper does not state the schedule; we use linear warmup
followed by cosine decay, and clip the gradient norm. The exported weights are the EMA
weights (with the usual decay warm-up); the in-loop evaluation uses the live weights.

Run with ``python -m romav2.train --help``; multi-GPU with ``torchrun --nproc_per_node N``.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from romav2.device import device
from romav2.features import FineFeatures
from romav2.romav2 import RELEASED_WEIGHTS_URL, RoMaV2
from romav2.train.data import (
    MATCHER_RESOLUTIONS,
    JsonPairDataset,
    MegaDepthDataset,
    PairDataset,
    mixture_weights,
    resize_batch,
)
from romav2.train.stages import MatcherStage, RefinerStage

logger = logging.getLogger(__name__)


@dataclass
class TrainCfg:
    stage: str = "matcher"  # "matcher" | "refiners"
    out: str = "runs/romav2"
    # data
    megadepth: str | None = None  # data root; adds MegaDepth with weight 1
    pairs: list[str] = field(default_factory=list)  # JSON indices, see JsonPairDataset
    val: str | None = None  # "megadepth" (Mega-1500 scenes) or a JSON index
    workers: int = 8
    # optimisation
    steps: int = 300_000
    batch_size: int = 128
    accum: int = 1  # gradient accumulation steps (effective batch = batch_size * accum * #GPUs)
    lr: float = 4e-4
    weight_decay: float = 0.01
    warmup: int = 1000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    bidirectional: bool = False  # supervise both A->B and B->A
    # initialisation
    from_scratch: bool = False  # random init of the trained parts (else: released weights)
    init: str | None = None  # full model state dict to start from (e.g. a stage-1 export)
    resume: str | None = None  # checkpoint.pt of a previous run
    # bookkeeping
    log_every: int = 50
    ckpt_every: int = 5000
    eval_every: int = 5000
    eval_batches: int = 50
    seed: int = 0
    wandb: bool = False


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------


def build_model(cfg: TrainCfg) -> RoMaV2:
    if cfg.from_scratch:
        # Random init of matcher and refiners; ImageNet VGG19 for the fine features (as RoMa).
        # The frozen DINOv3 backbone is taken from the released checkpoint (it is never trained).
        model = RoMaV2(
            RoMaV2.Cfg(pretrained=False, refiner_features=FineFeatures.Cfg(pretrained=True))
        )
        released = torch.hub.load_state_dict_from_url(RELEASED_WEIGHTS_URL, map_location=device)
        model.f.load_state_dict({k[len("f.") :]: v for k, v in released.items() if k.startswith("f.")})
    else:
        model = RoMaV2()
    if cfg.init is not None:
        model.load_state_dict(torch.load(cfg.init, map_location=device))
    return model


def build_stage(cfg: TrainCfg, model: RoMaV2) -> nn.Module:
    if cfg.stage == "matcher":
        stage = MatcherStage(model, bidirectional=cfg.bidirectional)
    elif cfg.stage == "refiners":
        stage = RefinerStage(model, bidirectional=cfg.bidirectional)
    else:
        raise ValueError(f"Unknown stage {cfg.stage}")
    for p in model.parameters():
        p.requires_grad_(False)
    for m in stage.trainable_modules().values():
        for p in m.parameters():
            p.requires_grad_(True)
    return stage


def build_datasets(cfg: TrainCfg) -> tuple[list[PairDataset], list[float]]:
    datasets, weights = [], []
    if cfg.megadepth is not None:
        datasets.append(MegaDepthDataset(cfg.megadepth, split="train"))
        weights.append(1.0)
    for index in cfg.pairs:
        d = JsonPairDataset(index)
        datasets.append(d)
        weights.append(d.weight)
    if not datasets:
        raise ValueError("No training data: pass --megadepth and/or --pairs")
    return datasets, weights


def build_val_dataset(cfg: TrainCfg) -> PairDataset | None:
    if cfg.val is None:
        return None
    if cfg.val == "megadepth":
        assert cfg.megadepth is not None, "--val megadepth needs --megadepth"
        return MegaDepthDataset(cfg.megadepth, split="test", augment=False)
    return JsonPairDataset(cfg.val, augment=False)


def ema_avg_fn(decay: float):
    """EMA update for ``AveragedModel`` with the usual decay warm-up
    min(decay, (1 + n) / (10 + n)) (as in TensorFlow's ExponentialMovingAverage / timm), so that
    short runs are not dominated by the initial weights. Equals ``decay`` after ~10k steps."""

    @torch.no_grad()
    def update(averaged: list[torch.Tensor], current: list[torch.Tensor], num_averaged: torch.Tensor):
        d = min(decay, (1 + num_averaged.item()) / (10 + num_averaged.item()))
        if torch.is_floating_point(averaged[0]):
            torch._foreach_lerp_(averaged, current, 1 - d)
        else:  # integer buffers (e.g. BatchNorm num_batches_tracked)
            for a, c in zip(averaged, current):
                a.copy_(c)

    return update


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def to_device(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def export_state_dict(model: RoMaV2, ema: AveragedModel) -> dict[str, torch.Tensor]:
    """Full ``RoMaV2`` state dict with the EMA weights of the trained modules swapped in."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for name, module in ema.module.items():
        for k, v in module.state_dict().items():
            state[f"{name}.{k}"] = v.detach().cpu()
    return state


@torch.no_grad()
def evaluate(stage: nn.Module, loader: DataLoader, max_batches: int) -> dict[str, float]:
    """Dense matching metrics (EPE in px, PCK@1/3/5 px) over covisible pixels, as in Table 6."""
    stage.eval()
    epe_sum, pck = 0.0, {1: 0.0, 3: 0.0, 5: 0.0}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = to_device(batch)
        H, W = batch["img_A"].shape[-2:]
        warp, _ = stage.predict(batch)
        to_px = torch.tensor((W / 2, H / 2), device=warp.device)
        epe = ((warp - batch["warp_A_to_B"]) * to_px).norm(dim=-1)
        covisible = batch["mask_A_to_B"][..., 0] > 0.99
        epe = epe[covisible]
        epe_sum += epe.sum().item()
        for t in pck:
            pck[t] += (epe < t).sum().item()
        n += epe.numel()
    stage.train()
    n = max(n, 1)
    return {"epe": epe_sum / n, **{f"pck_{t}px": v / n for t, v in pck.items()}}


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def train(cfg: TrainCfg, datasets: list[PairDataset] | None = None, weights: list[float] | None = None, val_dataset: PairDataset | None = None):
    """Train one stage. ``datasets`` / ``val_dataset`` override the ones built from ``cfg``."""
    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0
    torch.manual_seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    out = Path(cfg.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Config: {asdict(cfg)}")

    # model, stage, EMA, optimiser
    model = build_model(cfg)
    stage = build_stage(cfg, model)
    trainable = nn.ModuleDict(stage.trainable_modules())
    ema = AveragedModel(trainable, multi_avg_fn=ema_avg_fn(cfg.ema_decay), use_buffers=True)
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, cfg.warmup, cfg.steps)
    )
    step = 0
    if cfg.resume is not None:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        logger.info(f"Resumed from {cfg.resume} at step {step}")
    stage.train()
    num_params = sum(p.numel() for p in trainable.parameters())
    logger.info(f"Stage {cfg.stage}: {num_params / 1e6:.1f}M trainable parameters")
    runner = (
        DistributedDataParallel(stage, device_ids=[torch.cuda.current_device()])
        if distributed
        else stage
    )

    # data
    if datasets is None:
        datasets, weights = build_datasets(cfg)
    if val_dataset is None:
        val_dataset = build_val_dataset(cfg)
    sampler = WeightedRandomSampler(
        mixture_weights(datasets, weights),
        num_samples=(cfg.steps - step) * cfg.batch_size * cfg.accum,
        replacement=True,
        generator=torch.Generator().manual_seed(cfg.seed + rank),
    )
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None and is_main:
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=True)
    if cfg.wandb and is_main:
        import wandb

        wandb.init(project="roma-v2", name=out.name, config=asdict(cfg), resume="allow")

    def log(metrics: dict[str, float]):
        logger.info(f"step {step}: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
        if cfg.wandb and is_main:
            wandb.log(metrics, step=step)

    def checkpoint():
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": asdict(cfg),
            },
            out / "checkpoint.pt",
        )
        torch.save(export_state_dict(model, ema), out / f"romav2_{cfg.stage}.pt")

    # loop
    batches = iter(loader)
    t0 = time.perf_counter()
    while step < cfg.steps:
        running = {}
        for _ in range(cfg.accum):
            batch = to_device(next(batches))
            if cfg.stage == "matcher":
                batch = resize_batch(batch, random.choice(MATCHER_RESOLUTIONS))
            losses = runner(batch)
            loss = sum(losses.values())
            (loss / cfg.accum).backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item() / cfg.accum
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update_parameters(trainable)
        step += 1

        if step % cfg.log_every == 0 and is_main:
            elapsed = time.perf_counter() - t0
            log(
                {
                    "loss": sum(running.values()),
                    **running,
                    "grad_norm": grad_norm.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "steps_per_s": cfg.log_every / elapsed,
                }
            )
            t0 = time.perf_counter()
        if step % cfg.eval_every == 0 and val_loader is not None:
            log({f"val/{k}": v for k, v in evaluate(stage, val_loader, cfg.eval_batches).items()})
        if step % cfg.ckpt_every == 0 and is_main:
            checkpoint()

    if is_main:
        checkpoint()
        logger.info(f"Done. Weights (EMA) exported to {out / f'romav2_{cfg.stage}.pt'}")
    if distributed:
        dist.destroy_process_group()
