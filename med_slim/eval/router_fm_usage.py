"""
Histogram of per-FM router usage from a pretrained MoCo checkpoint to probe how pretrained router capture the importance of each FM during downstream tasks.

Uses the same path as linear probing to evaluate the router usage on downstream tasks:
  - default ``momentum`` encoder (optional ``--encoder {momentum,base}``)
  - eval FM set (K = a chosen eval subset)
  - per-slice router softmax over those K FMs, then fuse into the sequence encoder

Aggregates the averaged local router weights over valid slices. Baseline is `1/K` which is the uniform distribution.

Example:
    python -m med_slim.eval.router_fm_usage \
        --checkpoint /path/to/medslim-epoch2000.pth.tar \
        --num-batches 200 \
        --output router_usage.png
"""

from __future__ import annotations

import argparse
import math
import os
import torch
import yaml
from collections import defaultdict
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm
from accelerate import Accelerator

from med_slim.data.feat_dataset import UnlabeledFeatDataset, linear_classifier_collate_fn
from med_slim.eval.load_cobra import load_pretrained_cobra, resolve_eval_fm_ids


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_config(checkpoint: Path, config_arg: str | None) -> dict:
    if config_arg is not None:
        return _load_yaml(Path(config_arg))
    for candidate in (checkpoint.parent / "config.yaml", checkpoint.parent / "config.yml"):
        if candidate.exists():
            return _load_yaml(candidate)
    default = Path(__file__).resolve().parents[1] / "configs/pretrain.yml"
    if default.exists():
        print(f"No config next to checkpoint; falling back to {default}")
        return _load_yaml(default)
    raise FileNotFoundError(
        "Could not find config.yaml beside checkpoint. Pass --config explicitly."
    )


def _apply_checkpoint_metadata(cfg: dict, ckpt: dict) -> list[str]:
    """Prefer fm_id_order and router hyperparams stored in the checkpoint over config."""
    fm_names = ckpt.get("fm_id_order") or cfg["feat_dataset"]["model_name"]
    cfg["feat_dataset"]["model_name"] = list(fm_names)
    fm_choices = {m["name"]: m["embed_dim"] for m in cfg["model"]["slice_encoder_models"]}
    cobra_cfg = cfg["model"]["cobra"]
    cobra_cfg["input_dims"] = sorted(set(fm_choices[m] for m in fm_names))
    if ckpt.get("fm_pooling"):
        cobra_cfg["fm_pooling"] = ckpt["fm_pooling"]
    if ckpt.get("per_fm_adapter_mode"):
        cobra_cfg["per_fm_adapter_mode"] = ckpt["per_fm_adapter_mode"]
    if ckpt.get("fm_input_dims"):
        cobra_cfg["fm_input_dims"] = ckpt["fm_input_dims"]
    if ckpt.get("regional_tokens") is not None:
        cobra_cfg["regional_tokens"] = ckpt["regional_tokens"]
    if ckpt.get("physical_pe") is not None:
        cobra_cfg["physical_pe"] = ckpt["physical_pe"]
    if ckpt.get("sequence_encoder"):
        cfg["sequence_encoder"] = ckpt["sequence_encoder"]
        cobra_cfg["sequence_encoder"] = ckpt["sequence_encoder"]
    if ckpt.get("pooling"):
        cfg["pooling"] = ckpt["pooling"]
        cobra_cfg["pooling"] = ckpt["pooling"]
    for key in (
        "router_mode",
        "router_top_k",
        "router_temperature",
        "router_learnable_temperature",
        "router_use_fm_embedding",
        "router_use_fm_logit_bias",
        "router_use_fm_logit_scale",
    ):
        if key in ckpt:
            cobra_cfg[key] = ckpt[key]
    return list(fm_names)


class _TaggedUnlabeledDataset(Dataset):
    """Attach dataset/plane tags for stratified usage tables."""

    def __init__(self, dataset: UnlabeledFeatDataset, dataset_name: str, plane: str):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.plane = plane

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        item["dataset_name"] = self.dataset_name
        item["plane"] = self.plane
        item["study_id"] = item["sample_id"]
        return item


def _collate_fn(batch: list[dict]) -> dict:
    out = linear_classifier_collate_fn(batch)
    out["dataset_name"] = [b["dataset_name"] for b in batch]
    out["plane"] = [b["plane"] for b in batch]
    out["study_id"] = [b["study_id"] for b in batch]
    return out


def _build_fullset_dataset(cfg: dict, fm_names: list[str]) -> Dataset:
    """Inference loaders: all `fm_names` features for each study/plane."""
    feat_cfg = cfg["feat_dataset"]
    planes = feat_cfg.get("plane") or ["sagittal", "coronal", "axial"]
    parts: list[Dataset] = []
    for ds in feat_cfg["datasets"]:
        feat_dir = ds["feat_dir"]
        ds_name = ds.get("name", Path(feat_dir).parts[-3] if len(Path(feat_dir).parts) >= 3 else "unknown")
        for plane in planes:
            first_model = fm_names[0]
            plane_dir = os.path.join(feat_dir, first_model, "train", plane)
            if not os.path.isdir(plane_dir):
                print(f"Skipping missing plane cache: {plane_dir}")
                continue
            try:
                unlabeled = UnlabeledFeatDataset(
                    feat_dir=feat_dir,
                    slice_encoder_models=fm_names,
                    split="train",
                    view_plane=plane,
                )
            except FileNotFoundError as exc:
                print(f"Skipping ({exc})")
                continue
            parts.append(
                _TaggedUnlabeledDataset(unlabeled, dataset_name=str(ds_name), plane=plane)
            )
    if not parts:
        raise FileNotFoundError(
            "No feature caches found for LP full-set probe. Check feat_dataset.datasets / planes."
        )
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def _uniform_target(num_fms: int) -> float:
    return 1.0 / num_fms


def _print_table(fm_names: list[str], usage: torch.Tensor, k: int) -> None:
    target = _uniform_target(k)
    print(f"\n{'FM':<20} {'ID':>3} {'usage':>8} {'Difference compared to uniform':>12}")
    print("-" * 48)
    for idx, name in enumerate(fm_names):
        frac = usage[idx].item()
        print(f"{name:<20} {idx:>3} {frac:>8.4f} {frac - target:>+12.4f}")
    print(f"\nSum: {usage.sum().item():.4f}  |  Uniform baseline 1/K={target:.4f} (K={k})")


def _init_fm_usage_stats(num_fms: int) -> dict:
    return {
        "usage_sum": torch.zeros(num_fms, dtype=torch.float64),
        "count": 0,
    }


def _add_fm_usage(
    groups: dict[str, dict],
    keys: list[str],
    sample_usage: torch.Tensor,
    num_fms: int,
) -> None:
    sample_usage = sample_usage.detach().cpu().double()
    for key, usage in zip(keys, sample_usage):
        stats = groups[key]
        if not stats:
            stats.update(_init_fm_usage_stats(num_fms))
        stats["usage_sum"] += usage
        stats["count"] += 1


def _print_fm_usage(title: str, groups: dict[str, dict], fm_names: list[str]) -> None:
    if not groups:
        return
    print(f"\n{title}")
    print("-" * len(title))
    for key in sorted(groups):
        stats = groups[key]
        usage = stats["usage_sum"] / max(stats["count"], 1)
        top_vals, top_idx = torch.topk(usage, k=min(3, len(fm_names)))
        top_summary = ", ".join(
            f"{fm_names[idx]}={val:.4f}"
            for idx, val in zip(top_idx.tolist(), top_vals.tolist())
        )
        print(f"{key:<12} n_views={stats['count']:<5} top={top_summary}")


def _view_router_stats(
    cobra,
    features: list[torch.Tensor],
    seq_lens: torch.Tensor,
    fm_ids: torch.Tensor,
    physical_positions: torch.Tensor | None,
) -> dict:
    _ = cobra(
        features,
        seq_lengths=seq_lens,
        physical_positions=physical_positions,
        fm_ids=fm_ids,
    )
    stats = cobra._last_fm_stats
    if stats is None:
        raise RuntimeError("Cobra did not expose router stats (_last_fm_stats is None).")
    return stats


def _accumulate_router_stats(
    stats: dict,
    labels: dict[str, list[str]],
    accum: dict,
    num_fms: int,
) -> None:
    """Accumulate averaged per-slice router weights."""
    weights = stats["fm_weights_local"].detach()  # [B, D, K]
    fm_ids = stats["fm_ids"].detach().to(dtype=torch.long)  # [B, K] or broadcastable
    device = weights.device
    batch_size, _, k = weights.shape
    if k != num_fms:
        raise RuntimeError(f"Expected eval router with K={num_fms}, got K={k}.")
    if fm_ids.dim() == 1:
        fm_ids = fm_ids.unsqueeze(0).expand(batch_size, -1)
    elif fm_ids.shape[0] == 1 and batch_size != 1:
        fm_ids = fm_ids.expand(batch_size, -1)

    slice_mask = stats.get("slice_mask")
    if slice_mask is None:
        slice_mask = torch.ones(weights.shape[:2], device=device, dtype=torch.bool)
    else:
        slice_mask = slice_mask.to(device=device, dtype=torch.bool)
    valid = slice_mask.to(dtype=weights.dtype)
    valid_count = valid.sum().clamp_min(1.0)
    per_sample_valid = valid.sum(dim=1).clamp_min(1.0)  # [B]

    # Slice-averaged K-way weights; scatter keeps order-safe if the list is permuted.
    local_importance = (weights * valid.unsqueeze(-1)).sum(dim=1) / per_sample_valid[:, None]
    sample_usage = torch.zeros(batch_size, num_fms, device=device, dtype=weights.dtype)
    sample_usage.scatter_add_(1, fm_ids, local_importance)

    accum["usage_sum"] += sample_usage.sum(dim=0).double()
    accum["num_views"] += batch_size

    entropy = stats["fm_weight_entropy"].detach().to(device=device)
    accum["entropy_sum"] += (entropy * valid).sum().double()
    accum["max_weight_sum"] += (weights.max(dim=-1).values * valid).sum().double()
    top2 = weights.topk(k=min(2, k), dim=-1).values.sum(dim=-1)
    accum["top2_mass_sum"] += (top2 * valid).sum().double()
    accum["num_valid_slices"] += valid_count.double()

    top1_local = weights.argmax(dim=-1)
    top1_global = torch.gather(fm_ids, 1, top1_local)
    flat_top1 = top1_global[slice_mask].reshape(-1)
    if flat_top1.numel() > 0:
        accum["top1_slice_counts"].index_add_(
            0,
            flat_top1,
            torch.ones_like(flat_top1, dtype=accum["top1_slice_counts"].dtype),
        )

    sample_top1 = sample_usage.argmax(dim=1)
    accum["top1_sample_counts"].index_add_(
        0,
        sample_top1,
        torch.ones_like(sample_top1, dtype=accum["top1_sample_counts"].dtype),
    )

    sample_usage_cpu = sample_usage.cpu().double()
    _add_fm_usage(accum["by_dataset"], labels["dataset_name"], sample_usage_cpu, num_fms)
    _add_fm_usage(accum["by_plane"], labels["plane"], sample_usage_cpu, num_fms)
    for study_id, usage in zip(labels["study_id"], sample_usage_cpu):
        study_stats = accum["by_study"][study_id]
        if not study_stats:
            study_stats.update(_init_fm_usage_stats(num_fms))
        study_stats["usage_sum"] += usage
        study_stats["count"] += 1


def _print_sharpness(accum: dict, fm_names: list[str], k: int) -> None:
    num_valid = accum["num_valid_slices"].item()
    if num_valid <= 0:
        return
    entropy = (accum["entropy_sum"] / accum["num_valid_slices"]).item()
    max_weight = (accum["max_weight_sum"] / accum["num_valid_slices"]).item()
    top2_mass = (accum["top2_mass_sum"] / accum["num_valid_slices"]).item()
    norm_entropy = entropy / math.log(max(k, 2))

    print("\nRouter sharpness")
    print("------------------------------")
    print(f"Mean local entropy:     {entropy:.4f} (normalized={norm_entropy:.4f})")
    print(f"Mean max local weight:  {max_weight:.4f}")
    print(f"Mean top-2 local mass:  {top2_mass:.4f}")

    slice_total = accum["top1_slice_counts"].sum().clamp_min(1.0)
    sample_total = accum["top1_sample_counts"].sum().clamp_min(1.0)
    print("\nTop-1 rates")
    print("------------------------------------")
    print(f"{'FM':<20} {'top1_slice':>11} {'top1_sample':>12}")
    for idx, name in enumerate(fm_names):
        top1_slice = (accum["top1_slice_counts"][idx] / slice_total).item()
        top1_sample = (accum["top1_sample_counts"][idx] / sample_total).item()
        print(f"{name:<20} {top1_slice:>11.4f} {top1_sample:>12.4f}")


def _print_study_variance(accum: dict, fm_names: list[str]) -> None:
    if not accum["by_study"]:
        return
    study_usage = torch.stack([
        stats["usage_sum"] / max(stats["count"], 1)
        for stats in accum["by_study"].values()
    ])
    per_fm_std = study_usage.std(dim=0, unbiased=False)
    print("\nStudy-level variation")
    print("---------------------")
    print(
        "Mean per-FM std across processed studies: "
        f"{per_fm_std.mean().item():.4f}"
    )
    top_vals, top_idx = torch.topk(per_fm_std, k=min(3, len(fm_names)))
    top_summary = ", ".join(
        f"{fm_names[idx]}={val:.4f}"
        for idx, val in zip(top_idx.tolist(), top_vals.tolist())
    )
    print(f"FMs with the highest variation: {top_summary}")


def _save_plot(fm_names: list[str], usage: torch.Tensor, output: Path, k: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --output plots") from exc

    target = _uniform_target(k)
    fig, ax = plt.subplots(figsize=(max(8, len(fm_names) * 0.8), 4))
    x = range(len(fm_names))
    # Aggregation: per volume, mean softmax weight over valid slices; then mean over volumes.
    ax.bar(x, usage.cpu().numpy(), color="steelblue", label="FM-routing weight")
    ax.axhline(
        target,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"Uniform baseline ($1/K={target:.3f}$)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(fm_names, rotation=45, ha="right")
    ax.set_ylabel("FM-routing weight")
    ax.set_title("Per-FM router usage")
    ax.set_ylim(0, max(float(usage.max()) * 1.15, target * 1.5))
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart to {output}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Probe router FM usage.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to medslim-epoch*.pth.tar")
    parser.add_argument("--config", type=str, default=None, help="Pretrain config (default: beside checkpoint)")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["momentum", "base"],
        default="momentum",
        help="Which MoCo encoder to probe (default: momentum).",
    )
    parser.add_argument(
        "--fm-names",
        nargs="*",
        default=None,
        help=(
            "Optional FM subset to probe (names must appear in pretrain fm_id_order). "
            "Default: all pretrain FMs."
        ),
    )
    parser.add_argument("--num-batches", type=int, default=100, help="Batches to aggregate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size from config")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: min(4, train.num_workers))",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save bar-chart PNG")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = _resolve_config(checkpoint, args.config)
    pretrain_fm_names = _apply_checkpoint_metadata(cfg, ckpt)
    cobra_cfg = cfg["model"]["cobra"]

    if cobra_cfg.get("fm_pooling", "avg_pool") != "router":
        raise ValueError("Checkpoint/config is not fm_pooling='router'.")

    probe_fm_names = list(args.fm_names) if args.fm_names else list(pretrain_fm_names)
    eval_fm_ids = resolve_eval_fm_ids(
        "router", probe_fm_names, cfg, ckpt
    )
    k = len(probe_fm_names)
    if eval_fm_ids is None:
        raise RuntimeError("resolve_eval_fm_ids returned None for router pooling.")

    accelerator = Accelerator()
    device = torch.device(args.device)
    cobra = load_pretrained_cobra(
        checkpoint_path=str(checkpoint),
        accelerator=accelerator,
        model_config=cobra_cfg,
        encoder_type=args.encoder,
        fm_pooling="router",
        sequence_encoder=cfg.get("sequence_encoder") or cobra_cfg.get("sequence_encoder"),
        slice_pooling=cfg.get("pooling") or cobra_cfg.get("pooling") or cobra_cfg.get("slice_pooling"),
        pooling_target="post_embed",
        physical_pe=cobra_cfg.get("physical_pe", False),
    )
    cobra = cobra.to(device)
    cobra.eval()

    dataset = _build_fullset_dataset(cfg, probe_fm_names)
    batch_size = args.batch_size or min(64, cfg["train"].get("batch_size", 64))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(4, cfg["train"].get("num_workers", 4))
    dataloader_kwargs = {}
    if num_workers > 0:
        dataloader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate_fn,
        **dataloader_kwargs,
    )

    fm_ids = torch.as_tensor(eval_fm_ids, dtype=torch.long, device=device)
    accum = {
        "usage_sum": torch.zeros(k, device=device, dtype=torch.float64),
        "num_views": 0,
        "entropy_sum": torch.zeros((), device=device, dtype=torch.float64),
        "max_weight_sum": torch.zeros((), device=device, dtype=torch.float64),
        "top2_mass_sum": torch.zeros((), device=device, dtype=torch.float64),
        "num_valid_slices": torch.zeros((), device=device, dtype=torch.float64),
        "top1_slice_counts": torch.zeros(k, device=device, dtype=torch.float64),
        "top1_sample_counts": torch.zeros(k, device=device, dtype=torch.float64),
        "by_dataset": defaultdict(dict),
        "by_plane": defaultdict(dict),
        "by_study": defaultdict(dict),
    }
    n_batches = 0

    for batch in tqdm(loader, total=min(args.num_batches, len(loader)), desc="Aggregating LP usage"):
        if n_batches >= args.num_batches:
            break
        features = [
            f.to(device=device, dtype=next(cobra.parameters()).dtype) for f in batch["features"]
        ]
        seq_lens = batch["seq_lengths"].to(device=device, dtype=torch.long)
        physical_positions = batch.get("physical_positions")
        if physical_positions is not None:
            physical_positions = physical_positions.to(device=device, dtype=torch.float32)

        labels = {
            "dataset_name": batch["dataset_name"],
            "plane": batch["plane"],
            "study_id": batch["study_id"],
        }
        stats = _view_router_stats(cobra, features, seq_lens, fm_ids, physical_positions)
        _accumulate_router_stats(stats, labels, accum, k)
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("No batches processed.")

    mean_usage = (accum["usage_sum"] / max(accum["num_views"], 1)).cpu().float()
    router = cobra.fm_router
    temp = router.log_temperature.exp().item() if router is not None else float("nan")

    print(f"Checkpoint: {checkpoint}")
    print(f"Encoder: {args.encoder}")
    print(f"Probe mode: LP-aligned full-set inference (softmax over K={k})")
    print(f"Pretrain FM order (M={len(pretrain_fm_names)}): {pretrain_fm_names}")
    print(f"Probed FMs (K={k}): {probe_fm_names}")
    print(f"Global FM IDs: {eval_fm_ids}")
    print(f"Batches aggregated: {n_batches}")
    print(f"Volumes aggregated: {accum['num_views']}")
    print(f"Router temperature: {temp:.4f}")
    _print_table(probe_fm_names, mean_usage, k)
    _print_sharpness(accum, probe_fm_names, k)
    _print_fm_usage("Per-dataset routing usage", accum["by_dataset"], probe_fm_names)
    _print_fm_usage("Per-plane routing usage", accum["by_plane"], probe_fm_names)
    _print_study_variance(accum, probe_fm_names)

    if args.output:
        _save_plot(probe_fm_names, mean_usage, Path(args.output), k)


if __name__ == "__main__":
    main()
