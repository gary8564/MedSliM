"""
Histogram of per-FM router usage from a pretrained MoCo checkpoint.

Aggregates global marginal routing mass fractions over training batches (subset SSL).

Example:
    python -m med_slim.eval.router_fm_usage \
        --checkpoint /path/to/medslim-epoch2000.pth.tar \
        --num-batches 200 \
        --output router_usage.png
"""

from __future__ import annotations

import argparse
import math
import torch
import yaml
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from med_slim.data.feat_dataset import PrecomputedFeatPairDataset
from med_slim.model.ssl import MoCo


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
    cfg["feat_dataset"]["model_name"] = fm_names
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
    if ckpt.get("pooling"):
        cfg["pooling"] = ckpt["pooling"]
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
    return fm_names


def _build_moco(cfg: dict, device: torch.device) -> MoCo:
    cobra_cfg = cfg["model"]["cobra"]
    router_cfg = cfg.get("router", {}) or {}
    feat_cfg = cfg["feat_dataset"]
    num_fms = len(feat_cfg["model_name"])
    fm_input_dims = cobra_cfg.get("fm_input_dims")
    if fm_input_dims is None:
        name_to_dim = {m["name"]: m["embed_dim"] for m in cfg["model"]["slice_encoder_models"]}
        fm_input_dims = [name_to_dim[n] for n in feat_cfg["model_name"]]

    encoder_kwargs = {}
    if cfg.get("sequence_encoder", "mamba2") == "mamba2":
        encoder_kwargs["d_state"] = cobra_cfg.get("mamba_d_state", 128)
    else:
        encoder_kwargs["rotary_positional_encoding"] = cobra_cfg.get(
            "transformer_rotary_positional_encoding"
        )
        encoder_kwargs["norm_first"] = cobra_cfg.get("transformer_norm_first", True)
        encoder_kwargs["dim_feedforward"] = cobra_cfg.get(
            "transformer_dim_feedforward", 4 * cobra_cfg["embed_dim"]
        )
    if cfg.get("pooling", "abmil") == "abmil":
        encoder_kwargs["att_dim"] = cobra_cfg.get("attn_dim", 256)

    return MoCo(
        embed_dim=cobra_cfg["embed_dim"],
        contrast_dim=cobra_cfg["contrast_dim"],
        input_dims=cobra_cfg["input_dims"],
        num_heads=cobra_cfg["num_heads"],
        num_layers=cobra_cfg["num_layers"],
        T=cfg["train"]["temperature"],
        dropout=cobra_cfg["dropout"],
        sequence_encoder=cfg.get("sequence_encoder", "mamba2"),
        pooling=cfg.get("pooling", "abmil"),
        physical_pe=cobra_cfg.get("physical_pe", False),
        regional_tokens=cobra_cfg.get("regional_tokens", 0),
        fm_pooling=cobra_cfg.get("fm_pooling", "avg_pool"),
        num_fms=num_fms,
        per_fm_adapter_mode=cobra_cfg.get("per_fm_adapter_mode", "per_dim"),
        fm_input_dims=fm_input_dims,
        router_use_fm_embedding=cobra_cfg.get("router_use_fm_embedding", True),
        router_use_fm_logit_bias=cobra_cfg.get("router_use_fm_logit_bias", True),
        router_use_fm_logit_scale=cobra_cfg.get("router_use_fm_logit_scale", False),
        router_mode=cobra_cfg.get("router_mode", "soft"),
        router_top_k=cobra_cfg.get("router_top_k"),
        router_temperature=cobra_cfg.get("router_temperature", 1.0),
        router_learnable_temperature=cobra_cfg.get("router_learnable_temperature", False),
        router_load_balance_weight=router_cfg.get("load_balance_weight", 0.0),
        router_z_loss_weight=router_cfg.get("z_loss_weight", 0.0),
        router_entropy_weight=router_cfg.get("confidence_weight", 0.0),
        **encoder_kwargs,
    ).to(device)


def _build_dataset(cfg: dict, cache_in_memory: bool = False) -> PrecomputedFeatPairDataset:
    feat_cfg = cfg["feat_dataset"]
    ssl_cfg = cfg.get("ssl", {}) or {}
    cobra_cfg = cfg["model"]["cobra"]
    return PrecomputedFeatPairDataset(
        feat_dirs=feat_cfg["datasets"],
        slice_encoder_models=feat_cfg["model_name"],
        view_planes=feat_cfg["plane"],
        split="train",
        max_feature_dim=max(cobra_cfg["input_dims"]),
        num_target_slices=feat_cfg.get("num_target_slices", 32),
        cache_in_memory=cache_in_memory,
        use_packed=False,
        ssl_fm_mode=ssl_cfg.get("fm_mode", "pair"),
        fm_subset_size=ssl_cfg.get("fm_subset_size"),
        fm_subset_min_overlap=ssl_cfg.get("fm_subset_min_overlap", 1),
        fm_subset_max_overlap=ssl_cfg.get("fm_subset_max_overlap"),
    )


def _uniform_global_target(num_fms: int) -> float:
    return 1.0 / num_fms


def _print_table(fm_names: list[str], usage: torch.Tensor, num_fms: int) -> None:
    target = _uniform_global_target(num_fms)
    print(f"\n{'FM':<20} {'ID':>3} {'usage':>8} {'Difference compared to uniform':>10}")
    print("-" * 45)
    for idx, name in enumerate(fm_names):
        frac = usage[idx].item()
        print(f"{name:<20} {idx:>3} {frac:>8.4f} {frac - target:>+10.4f}")
    print(f"\nSum: {usage.sum().item():.4f}  |  Uniform global target: {target:.4f}")


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
    model: MoCo,
    x: torch.Tensor,
    dims: torch.Tensor,
    seq_lens: torch.Tensor,
    fm_ids: torch.Tensor,
    physical_positions: torch.Tensor | None,
) -> dict:
    _ = model.base_encoder(
        x,
        input_feature_dims=dims,
        seq_lengths=seq_lens,
        physical_positions=physical_positions,
        fm_ids=fm_ids,
    )
    stats = model.base_encoder._last_fm_stats
    if stats is None:
        raise RuntimeError("Base encoder did not expose router stats.")
    return stats


def _accumulate_router_stats(
    stats: dict,
    labels: dict[str, list[str]],
    accum: dict,
    num_fms: int,
) -> None:
    weights = stats["fm_weights_local"].detach()  # [B, D, K]
    fm_ids = stats["fm_ids"].detach().to(dtype=torch.long)  # [B, K]
    device = weights.device
    batch_size, _, subset_size = weights.shape
    slice_mask = stats.get("slice_mask")
    if slice_mask is None:
        slice_mask = torch.ones(weights.shape[:2], device=device, dtype=torch.bool)
    else:
        slice_mask = slice_mask.to(device=device, dtype=torch.bool)
    valid = slice_mask.to(dtype=weights.dtype)
    valid_count = valid.sum().clamp_min(1.0)
    per_sample_valid = valid.sum(dim=1).clamp_min(1.0)  # [B]

    # Slice-averaged local K-way importance per sample, then scatter back to global FM IDs.
    local_importance = (weights * valid.unsqueeze(-1)).sum(dim=1) / per_sample_valid[:, None]
    sample_usage = torch.zeros(batch_size, num_fms, device=device, dtype=weights.dtype)
    sample_usage.scatter_add_(1, fm_ids, local_importance)

    accum["usage_sum"] += sample_usage.sum(dim=0)
    accum["num_views"] += batch_size

    entropy = stats["fm_weight_entropy"].detach().to(device=device)
    accum["entropy_sum"] += (entropy * valid).sum()
    accum["max_weight_sum"] += (weights.max(dim=-1).values * valid).sum()
    top2 = weights.topk(k=min(2, subset_size), dim=-1).values.sum(dim=-1)
    accum["top2_mass_sum"] += (top2 * valid).sum()
    accum["num_valid_slices"] += valid_count

    top1_local = weights.argmax(dim=-1)  # [B, D]
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

    present_counts = per_sample_valid[:, None].expand_as(fm_ids).reshape(-1).to(
        dtype=accum["present_slice_counts"].dtype
    )
    accum["present_slice_counts"].index_add_(0, fm_ids.reshape(-1), present_counts)

    sample_usage_cpu = sample_usage.cpu().double()
    _add_fm_usage(accum["by_dataset"], labels["dataset_name"], sample_usage_cpu, num_fms)
    _add_fm_usage(accum["by_plane"], labels["plane"], sample_usage_cpu, num_fms)
    for study_id, usage in zip(labels["study_id"], sample_usage_cpu):
        study_stats = accum["by_study"][study_id]
        if not study_stats:
            study_stats.update(_init_fm_usage_stats(num_fms))
        study_stats["usage_sum"] += usage
        study_stats["count"] += 1


def _print_sharpness(accum: dict, fm_names: list[str], subset_size: int | None) -> None:
    num_valid = accum["num_valid_slices"].item()
    if num_valid <= 0:
        return
    entropy = (accum["entropy_sum"] / accum["num_valid_slices"]).item()
    max_weight = (accum["max_weight_sum"] / accum["num_valid_slices"]).item()
    top2_mass = (accum["top2_mass_sum"] / accum["num_valid_slices"]).item()
    norm_entropy = entropy / math.log(max(subset_size or 1, 2))

    print("\nRouter sharpness")
    print("----------------")
    print(f"Mean local entropy:     {entropy:.4f} (normalized={norm_entropy:.4f})")
    print(f"Mean max local weight:  {max_weight:.4f}")
    print(f"Mean top-2 local mass:  {top2_mass:.4f}")

    slice_total = accum["top1_slice_counts"].sum().clamp_min(1.0)
    sample_total = accum["top1_sample_counts"].sum().clamp_min(1.0)
    print("\nTop-1 and conditional win rates")
    print("-------------------------------")
    print(f"{'FM':<20} {'top1_slice':>11} {'top1_sample':>12} {'win_if_present':>15}")
    for idx, name in enumerate(fm_names):
        top1_slice = (accum["top1_slice_counts"][idx] / slice_total).item()
        top1_sample = (accum["top1_sample_counts"][idx] / sample_total).item()
        present = accum["present_slice_counts"][idx].clamp_min(1.0)
        win_if_present = (accum["top1_slice_counts"][idx] / present).item()
        print(f"{name:<20} {top1_slice:>11.4f} {top1_sample:>12.4f} {win_if_present:>15.4f}")


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
    print(f"Most variable FMs: {top_summary}")


def _save_plot(fm_names: list[str], usage: torch.Tensor, output: Path, num_fms: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --output plots") from exc

    target = _uniform_global_target(num_fms)
    fig, ax = plt.subplots(figsize=(max(8, len(fm_names) * 0.8), 4))
    x = range(len(fm_names))
    ax.bar(x, usage.cpu().numpy(), color="steelblue", label="routing mass fraction")
    ax.axhline(target, color="crimson", linestyle="--", linewidth=1.2, label=f"uniform 1/M={target:.3f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fm_names, rotation=45, ha="right")
    ax.set_ylabel("fraction of router mass")
    ax.set_title("Per-FM router usage (aggregated over batches)")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved histogram to {output}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Histogram router FM usage from a MoCo checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to medslim-epoch*.pth.tar")
    parser.add_argument("--config", type=str, default=None, help="Pretrain config (default: beside checkpoint)")
    parser.add_argument("--num-batches", type=int, default=100, help="Batches to aggregate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size from config")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: min(4, train.num_workers); increase for on-demand disk reads)",
    )
    parser.add_argument(
        "--cache-in-memory",
        action="store_true",
        help="Preload feature tensors into RAM before analysis. Faster for many batches, slower startup.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save bar-chart PNG")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = _resolve_config(checkpoint, args.config)
    fm_names = _apply_checkpoint_metadata(cfg, ckpt)

    if cfg["model"]["cobra"].get("fm_pooling", "avg_pool") != "router":
        raise ValueError("Checkpoint/config is not fm_pooling='router'.")

    device = torch.device(args.device)
    model = _build_moco(cfg, device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(
            "Warning: checkpoint/model key mismatch while loading. "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print(f"  first missing keys: {missing[:5]}")
        if unexpected:
            print(f"  first unexpected keys: {unexpected[:5]}")
    model.eval()

    dataset = _build_dataset(cfg, cache_in_memory=args.cache_in_memory)
    batch_size = args.batch_size or cfg["train"].get("batch_size", 256)
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(4, cfg["train"].get("num_workers", 4))
    dataloader_kwargs = {}
    if num_workers > 0:
        dataloader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        **dataloader_kwargs,
    )

    num_fms = len(fm_names)
    accum = {
        "usage_sum": torch.zeros(num_fms, device=device, dtype=torch.float64),
        "num_views": 0,
        "entropy_sum": torch.zeros((), device=device, dtype=torch.float64),
        "max_weight_sum": torch.zeros((), device=device, dtype=torch.float64),
        "top2_mass_sum": torch.zeros((), device=device, dtype=torch.float64),
        "num_valid_slices": torch.zeros((), device=device, dtype=torch.float64),
        "top1_slice_counts": torch.zeros(num_fms, device=device, dtype=torch.float64),
        "top1_sample_counts": torch.zeros(num_fms, device=device, dtype=torch.float64),
        "present_slice_counts": torch.zeros(num_fms, device=device, dtype=torch.float64),
        "by_dataset": defaultdict(dict),
        "by_plane": defaultdict(dict),
        "by_study": defaultdict(dict),
    }
    n_batches = 0

    for batch in tqdm(loader, total=min(args.num_batches, len(loader)), desc="Aggregating usage"):
        if n_batches >= args.num_batches:
            break
        x1 = batch["feats1"].to(device=device, dtype=torch.float32)
        x2 = batch["feats2"].to(device=device, dtype=torch.float32)
        dims1 = batch["orig_embed_dim1"].to(device=device, dtype=torch.long)
        dims2 = batch["orig_embed_dim2"].to(device=device, dtype=torch.long)
        seq_lens = batch["seq_len"].to(device=device, dtype=torch.long)
        fm_ids_1 = batch["fm_ids1"].to(device=device, dtype=torch.long)
        fm_ids_2 = batch["fm_ids2"].to(device=device, dtype=torch.long)
        physical_positions = batch.get("physical_positions")
        if physical_positions is not None:
            physical_positions = physical_positions.to(device=device, dtype=torch.float32)

        labels = {
            "dataset_name": batch.get("dataset_name", ["unknown"]),
            "plane": batch.get("plane", ["unknown"]),
            "study_id": batch.get("study_id", ["unknown"]),
        }

        stats1 = _view_router_stats(
            model, x1, dims1, seq_lens, fm_ids_1, physical_positions
        )
        _accumulate_router_stats(stats1, labels, accum, num_fms)
        stats2 = _view_router_stats(
            model, x2, dims2, seq_lens, fm_ids_2, physical_positions
        )
        _accumulate_router_stats(stats2, labels, accum, num_fms)
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("No batches processed.")

    mean_usage = (accum["usage_sum"] / max(accum["num_views"], 1)).cpu().float()
    subset_size = cfg.get("ssl", {}).get("fm_subset_size")
    router = model.base_encoder.fm_router
    temp = router.log_temperature.exp().item() if router is not None else float("nan")

    print(f"Checkpoint: {checkpoint}")
    print(f"FM order (M={num_fms}): {fm_names}")
    if subset_size is not None:
        print(f"Subset size (K): {subset_size}")
    print(f"Batches aggregated: {n_batches}")
    print(f"Views aggregated: {accum['num_views']}")
    print(f"Router temperature: {temp:.4f}")
    _print_table(fm_names, mean_usage, num_fms)
    _print_sharpness(accum, fm_names, subset_size)
    _print_fm_usage("Per-dataset routing usage", accum["by_dataset"], fm_names)
    _print_fm_usage("Per-plane routing usage", accum["by_plane"], fm_names)
    _print_study_variance(accum, fm_names)

    if args.output:
        _save_plot(fm_names, mean_usage, Path(args.output), num_fms)


if __name__ == "__main__":
    main()
