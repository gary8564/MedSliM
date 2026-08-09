"""
Adapted from:
[1] https://github.com/KatherLab/COBRA/blob/main/cobra/utils/mamba2.py
Lenz, Tim, Peter Neidlinger, Marta Ligero, Georg Wölflein, Marko van Treeck and Jakob Nikolas Kather. 
Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning.
2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 30807-30817, 2024.
[2] https://github.com/facebookresearch/moco-v3
Chen, Xinlei, Saining Xie and Kaiming He.
An Empirical Study of Training Self-Supervised Vision Transformers.
2021 IEEE/CVF International Conference on Computer Vision (ICCV) (2021): 9620-9629.
"""
import torch
import os
import builtins
import argparse
import yaml
import math
import wandb
from torch.utils.data import DataLoader
from jinja2 import Environment, FileSystemLoader
from pprint import pprint
from datetime import datetime
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from pathlib import Path

from med_slim.model.ssl import MoCo
from med_slim.data import PrecomputedFeatPairDataset
from med_slim.data.feat_dataset import ssl_packed_collate_fn

JOB_ID = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
CURR_TIME = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}_{JOB_ID}"


def _apply_cli_overrides(cfg: dict, args) -> None:
    """Override CLI flags into cfg so that the saved config.yaml matches the actual run."""
    feat_cfg = cfg.setdefault("feat_dataset", {})
    cobra_cfg = cfg.setdefault("model", {}).setdefault("cobra", {})

    if args.num_epochs is not None:
        cfg.setdefault("train", {})["num_epochs"] = args.num_epochs

    if args.model_names:
        fm_choices = {m["name"]: m["embed_dim"] for m in cfg["model"]["slice_encoder_models"]}
        unknown = [m for m in args.model_names if m not in fm_choices]
        if unknown:
            raise ValueError(f"Unknown model name(s) {unknown}. Known: {list(fm_choices.keys())}")
        feat_cfg["model_name"] = args.model_names
        cobra_cfg["input_dims"] = sorted(set(fm_choices[m] for m in args.model_names))

    if args.planes:
        feat_cfg["plane"] = args.planes

    if args.use_packed:
        feat_cfg["use_packed"] = True

    cobra_cfg["pooling"] = args.pooling
    cobra_cfg["sequence_encoder"] = args.sequence_encoder
    cobra_cfg["physical_pe"] = bool(
        args.physical_pe or cobra_cfg.get("physical_pe", False)
    )
    if args.regional_tokens is not None:
        if args.regional_tokens < 0:
            raise ValueError(f"--regional-tokens must be >= 0, got {args.regional_tokens}")
        cobra_cfg["regional_tokens"] = args.regional_tokens
        print(f"CLI override: model.cobra.regional_tokens={args.regional_tokens}")
    
    # FM fusion / subset / router CLI overrides
    cobra_cfg_dict = cfg["model"]["cobra"]
    if args.fm_pooling is not None:
        cobra_cfg_dict["fm_pooling"] = args.fm_pooling
        print(f"CLI override: fm_pooling={args.fm_pooling}")
    if args.per_fm_adapter_mode is not None:
        cobra_cfg_dict["per_fm_adapter_mode"] = args.per_fm_adapter_mode
        print(f"CLI override: per_fm_adapter_mode={args.per_fm_adapter_mode}")
    if args.router_mode is not None:
        cobra_cfg_dict["router_mode"] = args.router_mode
    if args.router_top_k is not None:
        cobra_cfg_dict["router_top_k"] = args.router_top_k
    if args.router_temperature is not None:
        cobra_cfg_dict["router_temperature"] = args.router_temperature

    if "ssl" not in cfg or cfg["ssl"] is None:
        cfg["ssl"] = {}
    if args.ssl_fm_mode is not None:
        cfg["ssl"]["fm_mode"] = args.ssl_fm_mode
        print(f"CLI override: ssl.fm_mode={args.ssl_fm_mode}")
    if args.fm_subset_size is not None:
        cfg["ssl"]["fm_subset_size"] = args.fm_subset_size
    if args.fm_subset_min_overlap is not None:
        cfg["ssl"]["fm_subset_min_overlap"] = args.fm_subset_min_overlap
    if args.fm_subset_max_overlap is not None:
        cfg["ssl"]["fm_subset_max_overlap"] = args.fm_subset_max_overlap

    if "router" not in cfg or cfg["router"] is None:
        cfg["router"] = {}
    if args.router_load_balance_weight is not None:
        cfg["router"]["load_balance_weight"] = args.router_load_balance_weight
    if args.router_z_loss_weight is not None:
        cfg["router"]["z_loss_weight"] = args.router_z_loss_weight
    if args.router_confidence_weight is not None:
        cfg["router"]["confidence_weight"] = args.router_confidence_weight
    

def _build_pretrain_run_name(cfg: dict, timestamp: str = CURR_TIME) -> str:
    """Compose the wandb run name from resolved config."""
    cobra_cfg = cfg["model"]["cobra"]
    datasets = cfg.get("feat_dataset", {}).get("datasets") or []
    names = [
        str(ds["name"]).strip()
        for ds in datasets
        if isinstance(ds, dict) and ds.get("name")
    ]
    dataset_tag = "-".join(names) if names else "unknown"
    return (
        f"{dataset_tag}-{cobra_cfg['sequence_encoder']}-{cobra_cfg['pooling']}"
        f"-{timestamp}"
    )


def _reset_router_log_temperature(model: MoCo, temperature: float) -> None:
    """Reinitialize router softmax temperature after curriculum weight transfer."""
    log_t = torch.log(torch.tensor(float(temperature)))
    for encoder_name in ("base_encoder", "momentum_encoder"):
        encoder = getattr(model, encoder_name, None)
        router = getattr(encoder, "fm_router", None) if encoder is not None else None
        if router is None or not hasattr(router, "log_temperature"):
            continue
        router.log_temperature.copy_(
            log_t.to(device=router.log_temperature.device, dtype=router.log_temperature.dtype)
        )


def validate_args(args) -> None:
    """Validate arguments parser."""
    valid_encoders = ["mamba2", "transformer"]
    valid_poolings = ["abmil", "cls"]
    
    if args.sequence_encoder not in valid_encoders:
        raise ValueError(f"Invalid sequence_encoder '{args.sequence_encoder}'. Must be one of {valid_encoders}")
    
    if args.pooling not in valid_poolings:
        raise ValueError(f"Invalid pooling '{args.pooling}'. Must be one of {valid_poolings}")
    
    if args.sequence_encoder == "mamba2" and args.pooling == "cls":
        raise ValueError(
            "Invalid configuration: mamba2 encoder cannot use 'cls' pooling. "
            "CLS token pooling requires transformer encoder. "
        )

def main(args, cfg):

    # Enable find_unused_parameters for DDP
    # Needed because COBRA uses nn.ModuleDict with embedding layers for each input dim, but only one is used per forward pass
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    if not accelerator.is_main_process:
        def print_pass(*_args, **_kwargs):
            pass
        builtins.print = print_pass

    pprint(cfg)

    # Initialize Weights & Biases on main process
    if accelerator.is_main_process:
        wandb.init(
            project="MedSliM-pretraining",
            name=_build_pretrain_run_name(cfg),
        )

    cobra_cfg = cfg["model"]["cobra"]
    sequence_encoder = cobra_cfg.get("sequence_encoder", args.sequence_encoder)
    pooling = cobra_cfg.get("pooling", args.pooling)
    physical_pe = cobra_cfg.get("physical_pe", False)
    regional_tokens = cobra_cfg.get("regional_tokens", 0)

    # FM fusion / router
    fm_pooling = cobra_cfg.get("fm_pooling", "avg_pool")
    ssl_cfg = cfg.get("ssl", {}) or {}
    router_cfg = cfg.get("router", {}) or {}
    ssl_fm_mode = ssl_cfg.get("fm_mode", "pair")
    use_packed = getattr(args, "use_packed", False)
    num_fms = len(cfg["feat_dataset"]["model_name"])
    log_per_fm_usage = bool(router_cfg.get("log_per_fm_usage", False))

    # Per-FM projection adapters: input dim per global FM id 
    # (ordered by feat_dataset.model_name which defines the global FM id order used by the router).
    per_fm_adapter_mode = cobra_cfg.get("per_fm_adapter_mode", "per_dim")
    name_to_dim = {m["name"]: m["embed_dim"] for m in cfg["model"]["slice_encoder_models"]}
    fm_input_dims = [name_to_dim[n] for n in cfg["feat_dataset"]["model_name"]]
    if fm_pooling == "router" and ssl_fm_mode != "subset":
        raise ValueError(
            f"fm_pooling='{fm_pooling}' requires ssl_fm_mode='subset' (FM-set views with fm_ids). "
            f"Got ssl_fm_mode='{ssl_fm_mode}'."
        )
    if ssl_fm_mode == "subset" and use_packed:
        raise NotImplementedError(
            "Packed subset FM mode is not supported yet. Use padded subset batches "
            "(use_packed=False) for fm_pooling router SSL."
        )

    # Build encoder-specific kwargs
    encoder_kwargs = {}
    
    if sequence_encoder == "mamba2":
        encoder_kwargs["d_state"] = cobra_cfg.get("mamba_d_state", 128)
    else:
        encoder_kwargs["rotary_positional_encoding"] = cobra_cfg.get("transformer_rotary_positional_encoding", None)
        encoder_kwargs["norm_first"] = cobra_cfg.get("transformer_norm_first", True)
        encoder_kwargs["dim_feedforward"] = cobra_cfg.get("transformer_dim_feedforward", 4 * cobra_cfg["embed_dim"])
    
    if pooling == "abmil":
        encoder_kwargs["att_dim"] = cobra_cfg.get("attn_dim", 256)
    
    # Build model
    print("Creating model...")
    if physical_pe:
        print("Physical positional encoding ENABLED (sinusoidal, normalized relative depth in [0, 1])")
    if regional_tokens > 0:
        print(
            f"Flattened regional-token pretraining ENABLED "
            f"(1 global + {regional_tokens} regional tokens per slice)"
        )
    print(f"FM fusion: fm_pooling={fm_pooling}, ssl_fm_mode={ssl_fm_mode}")
    model = MoCo(
        embed_dim=cobra_cfg["embed_dim"],
        contrast_dim=cobra_cfg["contrast_dim"],
        accelerator=accelerator,
        input_dims=cobra_cfg["input_dims"],
        num_heads=cobra_cfg["num_heads"],
        num_layers=cobra_cfg["num_layers"],
        T=cfg["train"]["temperature"],
        dropout=cobra_cfg["dropout"],
        sequence_encoder=sequence_encoder,
        pooling=pooling,
        physical_pe=physical_pe,
        regional_tokens=regional_tokens,
        fm_pooling=fm_pooling,
        num_fms=num_fms,
        per_fm_adapter_mode=per_fm_adapter_mode,
        fm_input_dims=fm_input_dims,
        router_use_fm_embedding=cobra_cfg.get("router_use_fm_embedding", True),
        router_use_fm_logit_bias=cobra_cfg.get("router_use_fm_logit_bias", True),
        router_use_fm_logit_scale=cobra_cfg.get("router_use_fm_logit_scale", False),
        router_mode=cobra_cfg.get("router_mode", "soft"),
        router_top_k=cobra_cfg.get("router_top_k", None),
        router_temperature=cobra_cfg.get("router_temperature", 1.0),
        router_learnable_temperature=cobra_cfg.get("router_learnable_temperature", False),
        router_load_balance_weight=router_cfg.get("load_balance_weight", 0.01),
        router_z_loss_weight=router_cfg.get("z_loss_weight", 0.0),
        router_entropy_weight=router_cfg.get("confidence_weight", 0.0),
        **encoder_kwargs,
    )

    model_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {model_params}")
    
    # Learning rate scaling rule 
    base_lr = float(cfg["train"]["learning_rate"])
    global_batch_size = cfg["train"]["batch_size"]
    scaled_lr = base_lr * (global_batch_size / 256.0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=cfg["train"]["weight_decay"])

    max_feature_dim = max(cobra_cfg["input_dims"])
    feat_cfg = cfg["feat_dataset"]
    feat_dirs = feat_cfg["datasets"]

    # If features were staged to local SSD, rewrite base paths
    local_base = os.environ.get("MEDSLIM_FEAT_BASE_OVERRIDE")
    if local_base:
        hpcwork_base = "/hpcwork/rwth1833/feat_caches"
        for ds in feat_dirs:
            ds["feat_dir"] = ds["feat_dir"].replace(hpcwork_base, local_base)
        print(f"Using local SSD feature cache: {local_base}")
    slice_encoder_models = feat_cfg["model_name"]
    view_planes = feat_cfg["plane"]
    use_packed = feat_cfg.get("use_packed", False)
    num_target_slices = feat_cfg.get("num_target_slices", 32)
    if use_packed:
        print("Packed mode: using raw variable-length sequences (no subsampling / zero-padding)")
    else:
        print(f"Pad-or-sample mode: all sequences will be sampled/padded to {num_target_slices} slices")
    if ssl_fm_mode == "subset":
        print(
            f"Subset FM SSL: fm_subset_size={ssl_cfg.get('fm_subset_size')}, "
            f"min_overlap={ssl_cfg.get('fm_subset_min_overlap', 1)}, "
            f"max_overlap={ssl_cfg.get('fm_subset_max_overlap')}"
        )
    cache_in_memory = args.cache_in_memory
    if cache_in_memory:
        print("Feature loading: cache_in_memory=True (full dataset preload into RAM)")
    else:
        print(
            "Feature loading: cache_in_memory=False (read from disk each batch; prefer staging to local NVMe on node's SSD)"
        )
    dataset = PrecomputedFeatPairDataset(
        feat_dirs=feat_dirs,
        slice_encoder_models=slice_encoder_models,
        view_planes=view_planes,
        split="train",
        max_feature_dim=max_feature_dim,
        num_target_slices=num_target_slices,
        cache_in_memory=cache_in_memory,
        use_packed=use_packed,
        ssl_fm_mode=ssl_fm_mode,
        fm_subset_size=ssl_cfg.get("fm_subset_size"),
        fm_subset_min_overlap=ssl_cfg.get("fm_subset_min_overlap", 1),
        fm_subset_max_overlap=ssl_cfg.get("fm_subset_max_overlap"),
    )

    if dataset.has_annotations:
        print(f"Semi-supervised mode: {dataset.num_labels} labels detected")
    else:
        print("Self-supervised mode: no annotations (pure InfoNCE)")

    # Optional: convert to SyncBatchNorm when training across processes for parity with DDP
    if accelerator.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    per_device_batch_size = max(1, int(global_batch_size / max(1, accelerator.num_processes)))
    print(f"Batch size per device: {per_device_batch_size}")
    print(f"Sequence mode: {'packed' if use_packed else 'random subsampling'}")

    # When CACHE_IN_MEMORY=false (required under 120G/1-GPU), fewer workers / less prefetch to avoid pushing over the limit.
    num_workers = cfg["train"]["num_workers"]
    prefetch_factor = 4 if cache_in_memory else 2
    num_workers = min(num_workers, 8) if not cache_in_memory else num_workers
    print(f"DataLoader: num_workers={num_workers}, prefetch_factor={prefetch_factor}")

    loader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0, # Keep workers alive between epochs when num_workers > 0
        prefetch_factor=prefetch_factor if num_workers > 0 else None, # Prefetch factor for the DataLoader
        collate_fn=ssl_packed_collate_fn if use_packed else None,
    )

    # Prepare with accelerator
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    
    # Load checkpoint if args.resume is provided
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location="cpu")
        # TODO: Remove this once all checkpoints are updated
        # Handle legacy checkpoint key names (mamba_enc -> seq_enc)
        # Old checkpoints used "mamba_enc" but new code uses generic "seq_enc"
        state_dict = checkpoint["state_dict"]
        
        # Handle legacy checkpoint key formats
        remapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Remap legacy mamba_enc -> seq_enc
            if "mamba_enc" in new_key:
                new_key = new_key.replace("mamba_enc", "seq_enc")
            remapped_state_dict[new_key] = value        
        if any("mamba_enc" in k for k in state_dict.keys()):
            print("Detected legacy checkpoint format, remapping keys: mamba_enc -> seq_enc")
        
        accelerator.unwrap_model(model).load_state_dict(remapped_state_dict)
        
        if args.curriculum:
            # Curriculum learning: load model weights only, reset optimizer and epoch
            # Use this when changing datasets (e.g., MRNet -> MRNet+fastMRI)
            print("Curriculum learning mode: loaded model weights, reset optimizer and epoch counter")
            print("Starting fresh training from epoch 0 with new dataset configuration")
            if args.router_temperature is not None:
                _reset_router_log_temperature(
                    accelerator.unwrap_model(model), args.router_temperature
                )
                print(
                    f"Curriculum: reset router log_temperature to {args.router_temperature}"
                )
        else:
            # Standard resume: continue training from saved state
            start_epoch = checkpoint.get("epoch", 0)
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Resuming training from epoch {start_epoch}")
    else:
        if args.resume:
            raise FileNotFoundError(f"No checkpoint found at '{args.resume}'")

    model.train()
    iters_per_epoch = len(loader)
    for e in tqdm(range(start_epoch, cfg["train"]["num_epochs"]), desc="MedSliM Pre-training...", disable=not accelerator.is_main_process):
        total_loss = 0.0

        for i, batch in enumerate(tqdm(loader, leave=False, disable=not accelerator.is_main_process)):
            curr_lr = adjust_learning_rate(optimizer, e + i / iters_per_epoch, scaled_lr, cfg)
            curr_m = adjust_moco_momentum(e + i / iters_per_epoch, cfg)

            optimizer.zero_grad(set_to_none=True)
            
            # Semi-supervised mode: pass labels when available (SupCon for labeled, InfoNCE for unlabeled)
            labels = batch.get("label", None)
            has_label = batch.get("has_label", None)
            if labels is not None:
                labels = labels.to(dtype=torch.float32)
            
            # Physical positions for sinusoidal PE (shared between views)
            phys_pos = batch.get("physical_positions")
            if phys_pos is not None:
                phys_pos = phys_pos.to(dtype=torch.float32)

            if use_packed:
                # Packed sequence mode
                x1 = batch["feats1"].to(dtype=torch.float32)
                x2 = batch["feats2"].to(dtype=torch.float32)
                cu_seqlens1 = batch["cu_seqlens1"].to(device=accelerator.device)
                cu_seqlens2 = batch["cu_seqlens2"].to(device=accelerator.device)
                max_seqlen1 = batch["max_seqlen1"]
                max_seqlen2 = batch["max_seqlen2"]
                sizes1 = batch["orig_embed_dim1"].to(dtype=torch.long)
                sizes2 = batch["orig_embed_dim2"].to(dtype=torch.long)
                seq_idx1 = batch["seq_idx1"].to(device=accelerator.device)
                seq_idx2 = batch["seq_idx2"].to(device=accelerator.device)
                
                with accelerator.autocast():
                    result = model(
                        x1, x2,
                        input_feature_dims_1=sizes1, input_feature_dims_2=sizes2,
                        m=curr_m,
                        use_packed=True,
                        cu_seqlens1=cu_seqlens1, cu_seqlens2=cu_seqlens2,
                        max_seqlen1=max_seqlen1, max_seqlen2=max_seqlen2,
                        seq_idx1=seq_idx1, seq_idx2=seq_idx2,
                        labels=labels,
                        has_label=has_label,
                        physical_positions=phys_pos,
                    )
            else:
            
                x1 = batch["feats1"].to(dtype=torch.float32)
                x2 = batch["feats2"].to(dtype=torch.float32)
                sizes1 = batch["orig_embed_dim1"].to(dtype=torch.long)
                sizes2 = batch["orig_embed_dim2"].to(dtype=torch.long)
                seq_lens = batch["seq_len"].to(dtype=torch.long)

                # Subset FM mode: pass global FM IDs so fusion/router/load-balance use stable identities
                fm_ids_1 = batch["fm_ids1"].to(device=accelerator.device, dtype=torch.long) if "fm_ids1" in batch else None
                fm_ids_2 = batch["fm_ids2"].to(device=accelerator.device, dtype=torch.long) if "fm_ids2" in batch else None

                with accelerator.autocast():
                    result = model(
                        x1, x2, 
                        input_feature_dims_1=sizes1, input_feature_dims_2=sizes2, 
                        seq_lengths=seq_lens, 
                        m=curr_m,
                        labels=labels,
                        has_label=has_label,
                        physical_positions=phys_pos,
                        fm_ids_1=fm_ids_1,
                        fm_ids_2=fm_ids_2,
                    )
            
            if isinstance(result, dict):
                loss = result["loss"]
                fm_usage = result.get("fm_usage")
                loss_components = {
                    k: v.item()
                    for k, v in result.items()
                    if k not in ("loss", "fm_usage") and torch.is_tensor(v) and v.ndim == 0
                }
            else:
                loss = result
                loss_components = {}
                fm_usage = None

            # NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf loss detected at epoch {e+1} with iter {i}.")
            
            accelerator.backward(loss)
            # Gradient clipping to prevent exploding gradients
            accelerator.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            loss_val = loss.detach().item()
            total_loss += loss_val
            
            # Log iteration-level metrics to WandB (every 10 steps to reduce overhead)
            if accelerator.is_main_process and i % 10 == 0:
                global_step = e * iters_per_epoch + i
                log_dict = {
                    "train/step_loss": loss_val,
                    "train/lr": curr_lr,
                    "train/momentum": curr_m,
                }
                for comp_name, comp_val in loss_components.items():
                    log_dict[f"train/{comp_name}"] = comp_val
                if (
                    log_per_fm_usage
                    and fm_pooling == "router"
                    and fm_usage is not None
                ):
                    for fm_idx, fm_name in enumerate(slice_encoder_models):
                        if fm_idx < fm_usage.shape[0]:
                            log_dict[f"train/fm_usage/{fm_name}"] = fm_usage[fm_idx].item()
                if fm_pooling == "router":
                    fm_router = accelerator.unwrap_model(model).base_encoder.fm_router
                    if fm_router is not None:
                        log_dict["train/router_temperature"] = (
                            fm_router.log_temperature.exp().item()
                        )
                if has_label is not None:
                    num_labeled = has_label.sum().item()
                    log_dict["train/labeled_ratio"] = num_labeled / has_label.shape[0]
                wandb.log(log_dict, step=global_step)

        # Synchronization barrier: ensure all ranks finished the epoch before logging/checkpointing
        # This helps catch GPU desync issues early rather than hanging during the next epoch
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {e+1}; loss: {avg_loss:.4f}; lr: {curr_lr:.5f}")
            wandb.log({
                "train/epoch": e + 1,
                "train/epoch_loss": avg_loss,
            }, step=e * iters_per_epoch + iters_per_epoch)
            if (e + 1) % 200 == 0:
                state = {
                    "epoch": e + 1,
                    "state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sequence_encoder": sequence_encoder,
                    "pooling": pooling,
                    "physical_pe": physical_pe,
                    "regional_tokens": regional_tokens,
                    "fm_pooling": fm_pooling,
                    "num_fms": num_fms,
                    "per_fm_adapter_mode": per_fm_adapter_mode,
                    "fm_input_dims": fm_input_dims,
                    "fm_id_order": cfg["feat_dataset"]["model_name"],
                    "ssl_fm_mode": ssl_fm_mode,
                    "fm_subset_size": ssl_cfg.get("fm_subset_size"),
                    "fm_subset_min_overlap": ssl_cfg.get("fm_subset_min_overlap", 1),
                    "fm_subset_max_overlap": ssl_cfg.get("fm_subset_max_overlap"),
                    "router_mode": cobra_cfg.get("router_mode", "soft"),
                    "router_top_k": cobra_cfg.get("router_top_k", None),
                    "router_temperature": cobra_cfg.get("router_temperature", 1.0),
                    "router_learnable_temperature": cobra_cfg.get("router_learnable_temperature", False),
                    "router_use_fm_embedding": cobra_cfg.get("router_use_fm_embedding", True),
                    "router_use_fm_logit_bias": cobra_cfg.get("router_use_fm_logit_bias", True),
                    "router_use_fm_logit_scale": cobra_cfg.get("router_use_fm_logit_scale", False),
                    "router_load_balance_weight": router_cfg.get("load_balance_weight", 0.01),
                    "router_z_loss_weight": router_cfg.get("z_loss_weight", 0.0),
                    "router_confidence_weight": router_cfg.get("confidence_weight", 0.0),
                }
                ckpt_name = f"medslim-epoch{e+1}.pth.tar"
                torch.save(
                    state,
                    os.path.join(cfg["train"]["save_ckpt_path"], ckpt_name),
                )


def adjust_learning_rate(optimizer, epoch, scaled_base_lr, cfg):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg["train"]["warmup_epochs"]:
        lr = scaled_base_lr * epoch / cfg["train"]["warmup_epochs"]
    else:
        lr = scaled_base_lr * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - cfg["train"]["warmup_epochs"]) / (cfg["train"]["num_epochs"] - cfg["train"]["warmup_epochs"]))
        )
        
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def adjust_moco_momentum(epoch, cfg):
    """Adjust moco momentum based on current epoch"""
    m = 1.0 - 0.5 * (1.0 + math.cos(math.pi * epoch / cfg["train"]["num_epochs"])) * (
        1.0 - cfg["train"]["momentum"]
    )
    return m

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSliM-pretraining.")
    parser.add_argument(
        "-c", "--config", type=str, default="../configs/pretrain.yml", help="Path to the config file"
    )
    parser.add_argument(
        "--planes", 
        nargs='*',  
        type=str,   
        help='A list of planes to preprocess.',
    )
    parser.add_argument(
        "--model-names",
        nargs='*',
        type=str,
        help=(
            "Override slice encoder model names from config. "
            "Example: --model-names dinov2 rad-dino mri-core. "
        ),
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="Path to latest checkpoint",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override train.num_epochs from config.",
    )
    parser.add_argument(
        "--sequence-encoder",
        type=str,
        choices=["mamba2", "transformer"],
        default="mamba2",
        help="Sequence encoder model: 'mamba2' (default) or 'transformer'",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["abmil", "cls"],
        default="abmil",
        help="Slice pooling method: 'abmil' (default) or 'cls'. Note: 'cls' requires transformer encoder.",
    )
    parser.add_argument(
        "--physical-pe",
        action="store_true",
        help=(
            "Enable sinusoidal positional encoding keyed on normalized relative "
            "slice depth in [0, 1] (fraction through the volume)."
        ),
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help=(
            "Curriculum learning mode: Load model weights from --resume checkpoint but "
            "reset optimizer state and epoch counter. Use this when increasing datasets "
            "(e.g., pretrain on MRNet, then continue with MRNet+fastMRI). "
        ),
    )
    parser.add_argument(
        "--use-packed",
        action="store_true",
        help=(
            "Use packed sequences instead of padded sequences for pretraining. "
            "Packed mode concatenates variable-length sequences and uses cu_seqlens "
            "to track boundaries, avoiding padding waste. "
            "Requires Mamba2 (seq_idx) or Transformer with FlashAttention (varlen_attn)."
        ),
    )
    parser.add_argument(
        "--regional-tokens",
        type=int,
        default=None,
        help=(
            "Flatten regional crop tokens into the sequence dimension. "
            "Example: --regional-tokens 4 expects global + 2x2 regional tokens "
            "per slice and creates num_slices*5 sequence tokens. "
            "Overrides config model.cobra.regional_tokens."
        ),
    )
    # FM fusion / router arguments
    parser.add_argument(
        "--fm-pooling",
        type=str,
        choices=["avg_pool", "router"],
        default=None,
        help="FM fusion mode. Overrides config model.cobra.fm_pooling.",
    )
    parser.add_argument(
        "--per-fm-adapter-mode",
        type=str,
        choices=["per_dim", "per_fm_id"],
        default=None,
        help=(
            "Projection-adapter keying for FM-set paths. 'per_dim' (shared by input dim) or "
            "'per_fm_id' (one adapter per FM). Overrides config model.cobra.per_fm_adapter_mode."
        ),
    )
    parser.add_argument(
        "--ssl-fm-mode",
        type=str,
        choices=["pair", "subset"],
        default=None,
        help="Positive-pair construction. 'pair' (cross-FM baseline) or 'subset'. Overrides config ssl.fm_mode.",
    )
    parser.add_argument("--fm-subset-size", type=int, default=None, help="FMs per view in subset mode.")
    parser.add_argument("--fm-subset-min-overlap", type=int, default=None, help="Min shared FMs between views.")
    parser.add_argument("--fm-subset-max-overlap", type=int, default=None, help="Max shared FMs between views.")
    parser.add_argument("--router-mode", type=str, choices=["soft", "topk"], default=None, help="Router mode.")
    parser.add_argument("--router-top-k", type=int, default=None, help="Top-k FMs when router_mode=topk.")
    parser.add_argument("--router-temperature", type=float, default=None, help="Router softmax temperature.")
    parser.add_argument(
        "--router-load-balance-weight", type=float, default=None,
        help="Weight for router load-balance loss. Overrides config router.load_balance_weight.",
    )
    parser.add_argument(
        "--router-z-loss-weight", type=float, default=None,
        help="Weight for router z-loss. Overrides config router.z_loss_weight.",
    )
    parser.add_argument(
        "--router-confidence-weight", type=float, default=None,
        help="Weight for router confidence (low-entropy) loss. Overrides config router.confidence_weight.",
    )
    parser.add_argument(
        "--cache-in-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preload all feature tensors into RAM at init (default). "
            "Use --no-cache-in-memory when the dataset exceeds the host-RAM budget "
            "(e.g. large pretraining datasets under the GPU memory quota limits)."
        ),
    )
    args = parser.parse_args()
    
    # Validate arg parser
    validate_args(args)
    
    # Get the directory of the current script
    curr_dir = Path(__file__).resolve().parent
    config_path = os.path.join(curr_dir, args.config)

    with open(config_path, "r") as f:
        cfg_data = yaml.safe_load(f)

    template_env = Environment(loader=FileSystemLoader(searchpath="./"))
    template = template_env.from_string(yaml.dump(cfg_data, default_flow_style=False))
    # Render the template with the values from the config_data
    cfg = yaml.safe_load(template.render(**cfg_data))
    
    _apply_cli_overrides(cfg, args)

    # Cross-FM contrastive learning requires at least 2 foundation models
    if len(cfg["feat_dataset"]["model_name"]) < 2:
        raise ValueError(
            f"Cross-FM contrastive learning requires at least 2 foundation models, "
            f"but only got {cfg['feat_dataset']['model_name']}. "
            f"Add more models via config feat_dataset.model_name or --model-names."
        )
    
    # If standard resume, use the same directory as the loaded checkpoint;
    # otherwise (fresh start/curriculum learning) create a new saved checkpoint folder
    if args.resume and not args.curriculum:
        save_dir = str(Path(args.resume).parent)
        print(f"Resuming from checkpoint, saving to existing directory: {save_dir}")
    else:
        save_dir = f"{cfg['train']['save_ckpt_path']}/{CURR_TIME}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
        if args.curriculum:
            print(f"Curriculum learning: saving new checkpoints to separate directory: {save_dir}")
    
    # Update config with the resolved save directory
    cfg["train"]["save_ckpt_path"] = save_dir
    
    main(args, cfg)