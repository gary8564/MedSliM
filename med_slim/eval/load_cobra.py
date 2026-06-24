import torch
import yaml
import logging
import os
from pathlib import Path
from accelerate import Accelerator
from typing import Any, Dict, Optional, Tuple

from med_slim.model.sequence_encoder.cobra import Cobra
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


def resolve_eval_fm_ids(
    fm_pooling: str,
    model_names: list[str],
    pretrain_cfg: Dict,
    pretrain_state: Optional[Dict],
) -> Optional[list[int]]:
    """Map eval FM names to the global FM IDs used when training the router."""
    if fm_pooling != "router":
        return None

    fm_id_order = None
    if pretrain_state is not None:
        fm_id_order = pretrain_state.get("fm_id_order")
    if fm_id_order is None:
        fm_id_order = pretrain_cfg.get("feat_dataset", {}).get("model_name")
    if not fm_id_order:
        raise ValueError(
            "fm_pooling='router' requires the pretraining FM order to recover global FM IDs. "
            "Expected checkpoint['fm_id_order'] or pretrain config feat_dataset.model_name."
        )

    name_to_id = {name: idx for idx, name in enumerate(fm_id_order)}
    missing = [name for name in model_names if name not in name_to_id]
    if missing:
        raise ValueError(
            "Requested FM models are not present in the router pretraining FM order: "
            f"{missing}. Available FMs: {fm_id_order}."
        )
    return [name_to_id[name] for name in model_names]


def _build_cobra(
    model_config: Dict,
    sequence_encoder: str,
    slice_pooling: str,
    fm_pooling: str,
    pooling_target: Optional[str] = None,
    raw_output_dim: Optional[int] = None,
    physical_pe: Optional[bool] = None,
    regional_tokens: Optional[int] = None,
    num_fms: Optional[int] = None,
    per_fm_adapter_mode: str = "per_dim",
    fm_input_dims: Optional[list] = None,
    router_kwargs: Optional[Dict[str, Any]] = None,
) -> Cobra:
    """Construct a COBRA model in inference mode from model configuration."""
    embed_dim = model_config["embed_dim"]
    encoder_kwargs: Dict[str, Any] = {}
    if physical_pe is None:
        physical_pe = model_config.get("physical_pe", False)
    if regional_tokens is None:
        regional_tokens = model_config.get("regional_tokens", 0)

    if sequence_encoder == "mamba2":
        encoder_kwargs["d_state"] = model_config.get("mamba_d_state", 128)
    elif sequence_encoder == "transformer":
        encoder_kwargs["rotary_positional_encoding"] = model_config["transformer_rotary_positional_encoding"]
        encoder_kwargs["norm_first"] = model_config["transformer_norm_first"]
        encoder_kwargs["dim_feedforward"] = model_config.get(
            "transformer_dim_feedforward",
            4 * embed_dim,
        )

    if slice_pooling == "abmil":
        encoder_kwargs["att_dim"] = model_config["attn_dim"]

    # Router config (only relevant when fm_pooling == "router").
    if router_kwargs:
        encoder_kwargs.update(router_kwargs)

    num_layers = model_config.get("num_layers", model_config.get("num_mamba_layers"))

    return Cobra(
        input_dims=model_config["input_dims"],
        embed_dim=embed_dim,
        contrast_dim=model_config["contrast_dim"],
        num_heads=model_config["num_heads"],
        num_layers=num_layers,
        dropout=model_config["dropout"],
        mode="inference",
        sequence_encoder=sequence_encoder,
        fm_pooling=fm_pooling,
        slice_pooling=slice_pooling,
        pooling_target=pooling_target,
        raw_output_dim=raw_output_dim,
        physical_pe=physical_pe,
        regional_tokens=regional_tokens,
        num_fms=num_fms,
        per_fm_adapter_mode=per_fm_adapter_mode,
        fm_input_dims=fm_input_dims,
        **encoder_kwargs,
    )


def _resolve_per_fm_adapter(meta: Dict[str, Any], cobra_weights: Dict[str, Any], prefix: str = ""):
    """
    Resolve (per_fm_adapter_mode, fm_input_dims) from checkpoint metadata or weights.

    Prefers saved metadata. 
    Fallback to inferring from FM-keyed adapter weights when metadata is missing 
    (e.g. older checkpoints saved before the metadata was added).
    (``<prefix>embed_fm.<id>.head.1.weight`` has shape ``[embed_dim, input_dim]``).

    Args:
        meta: Metadata.
        cobra_weights: The cobra encoder state dict.
        prefix: Key prefix on the cobra weights.
    """
    fm_input_dims = meta.get("fm_input_dims")
    embed_fm_prefix = f"{prefix}embed_fm."
    head_suffix = ".head.1.weight"

    has_per_fm_weights = any(k.startswith(embed_fm_prefix) for k in cobra_weights.keys())
    if not has_per_fm_weights:
        # No FM-keyed adapters in this checkpoint; use dim-keyed adapters.
        return "per_dim", None

    if fm_input_dims is None:
        dims_by_id: Dict[int, int] = {}
        for key, tensor in cobra_weights.items():
            if key.startswith(embed_fm_prefix) and key.endswith(head_suffix):
                fm_id = int(key[len(embed_fm_prefix):].split(".")[0])
                dims_by_id[fm_id] = int(tensor.shape[1])
        fm_input_dims = [dims_by_id[i] for i in sorted(dims_by_id)]
    return "per_fm_id", fm_input_dims


def load_pretrained_cobra(
    checkpoint_path: str, 
    accelerator: Accelerator, 
    model_config: Dict,
    encoder_type: str = "momentum",
    fm_pooling: str = "avg_pool",
    sequence_encoder: Optional[str] = None,
    slice_pooling: Optional[str] = None,
    pooling_target: Optional[str] = None,
    raw_output_dim: Optional[int] = None,
    physical_pe: Optional[bool] = None,
) -> Cobra:
    """
    Load the COBRA model from a pretrained checkpoint.

    Parameters:
    - checkpoint_path (str): Path to the model checkpoint file.
    - accelerator (Accelerator): HuggingFace Accelerator.
    - model_config (Dict): Dictionary containing the model configuration from pretrain config.
    - encoder_type (str): Choose between "base" and "momentum" encoder for downstream tasks. Default is "momentum".
    - fm_pooling (str): Feature aggregation method. Default is "avg_pool".
    - sequence_encoder (str, optional): Override sequence encoder type. If None, uses checkpoint or defaults to "mamba2".
    - slice_pooling (str, optional): Override slice pooling type. If None, uses saved checkpoint.
    - pooling_target (str, optional): Which representation to pool at inference ('post_encoder', 'post_embed', 'raw'). 
     If None, Cobra resolves a safe default from mode and slice pooling.
    - raw_output_dim (int, optional): FM embedding dimension, required when pooling_target='raw'.

    Returns:
    - Cobra: The loaded COBRA model in inference mode.
    
    Raises:
    - FileNotFoundError: If the checkpoint file is not found.
    - ValueError: If the checkpoint format is invalid.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    
    state_dict = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)
    if sequence_encoder is None:
        sequence_encoder = state_dict.get("sequence_encoder", "mamba2")  # Default for older checkpoints
    if slice_pooling is None:
        slice_pooling = state_dict.get("pooling", "abmil")  # Default for older checkpoints
    if physical_pe is None:
        physical_pe = state_dict.get("physical_pe", model_config.get("physical_pe", False))
    regional_tokens = int(
        state_dict.get("regional_tokens", model_config.get("regional_tokens", 0))
    )
    # Extract encoder weights from checkpoint
    if "state_dict" not in list(state_dict.keys()):
        raise ValueError(f"`state_dict` key not found in saved model checkpoint {checkpoint_path}.")
    
    chkpt = state_dict["state_dict"]

    # Handle legacy checkpoint key names (mamba_enc -> seq_enc)
    has_legacy_mamba = any("mamba_enc" in k for k in chkpt.keys())
    if has_legacy_mamba:
        logger.info("Detected legacy checkpoint format, remapping keys: mamba_enc -> seq_enc")
    
    cobra_weights = {}
    for k, v in chkpt.items():
        if f"{encoder_type}_encoder" in k and f"{encoder_type}_encoder.proj" not in k:
            # Remove encoder prefix (e.g., "base_encoder." or "momentum_encoder.")
            new_key = k.split(f"{encoder_type}_encoder.")[-1]
            # Remap legacy key names
            if has_legacy_mamba:
                new_key = new_key.replace("mamba_enc", "seq_enc")
            cobra_weights[new_key] = v
    
    if len(cobra_weights) == 0:
        raise ValueError(f"No {encoder_type} encoder weights found in checkpoint.")

    # FM fusion: detect router from the encoder weights. 
    num_fms = state_dict.get("num_fms")
    router_kwargs = None
    has_router_weights = any(k.startswith("fm_router.") for k in cobra_weights.keys())
    if fm_pooling == "router":
        if not has_router_weights:
            raise ValueError(
                "fm_pooling='router' requested but checkpoint has no fm_router.* weights."
            )
        if num_fms is None:
            embed_key = "fm_router.fm_embed.weight"
            bias_key = "fm_router.fm_logit_bias.weight"
            if embed_key in cobra_weights:
                num_fms = int(cobra_weights[embed_key].shape[0])
            elif bias_key in cobra_weights:
                num_fms = int(cobra_weights[bias_key].shape[0])
            else:
                raise ValueError("Cannot infer num_fms for router from checkpoint.")
        router_kwargs = dict(
            router_use_fm_embedding=state_dict.get("router_use_fm_embedding", "fm_router.fm_embed.weight" in cobra_weights),
            router_use_fm_logit_bias=state_dict.get("router_use_fm_logit_bias", "fm_router.fm_logit_bias.weight" in cobra_weights),
            router_use_fm_logit_scale=state_dict.get("router_use_fm_logit_scale", "fm_router.fm_logit_scale.weight" in cobra_weights),
            router_mode=state_dict.get("router_mode", "soft"),
            router_top_k=state_dict.get("router_top_k", None),
            router_temperature=state_dict.get("router_temperature", 1.0),
            router_learnable_temperature=state_dict.get("router_learnable_temperature", False),
        )
    elif fm_pooling == "avg_pool" and has_router_weights:
        logger.info("Checkpoint contains fm_router weights but fm_pooling='avg_pool' override is active (router bypassed).")

    # Per-FM projection adapters: rebuild FM-keyed adapters when the checkpoint used them.
    per_fm_adapter_mode, fm_input_dims = _resolve_per_fm_adapter(state_dict, cobra_weights)

    logger.info(
        f"Loading COBRA with sequence_encoder={sequence_encoder}, slice_pooling={slice_pooling}, "
        f"fm_pooling={fm_pooling}, pooling_target={pooling_target}, physical_pe={physical_pe}, "
        f"regional_tokens={regional_tokens}, num_fms={num_fms}, "
        f"per_fm_adapter_mode={per_fm_adapter_mode}"
    )

    model = _build_cobra(
        model_config, sequence_encoder, slice_pooling, fm_pooling, pooling_target,
        raw_output_dim, physical_pe, regional_tokens, num_fms=num_fms, per_fm_adapter_mode=per_fm_adapter_mode,
        fm_input_dims=fm_input_dims, router_kwargs=router_kwargs,
    )
    logger.info(f"Inference pooling_target={model.pooling_target}")

    # strict=False: proj layer exists in pretrained model but excluded from checkpoint (not used in inference mode)
    model.load_state_dict(cobra_weights, strict=False)
    logger.info(f"{encoder_type.capitalize()} COBRA model loaded successfully.")
    
    return model


def load_cobra_from_experiment(
    experiment_dir: str,
    accelerator: Accelerator,
    fold: Optional[int] = None,
) -> Tuple[Cobra, Dict[str, Any]]:
    """
    Load a COBRA model from a linear-probing saved checkpoints and configuration.

    When 'fold' is given the checkpoint is loaded from '{experiment_dir}/fold_{fold}/ckpt/classifier.pt' instead of the
    default '{experiment_dir}/ckpt/classifier.pt'.
    """
    experiment_dir = Path(experiment_dir)

    # Load experiment config
    config_path = experiment_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yml not found in {experiment_dir}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cobra_cfg = cfg.get("cobra_config")
    if cobra_cfg is None:
        raise ValueError(
            "cobra_config not found in config.yml. "
            "Re-run linear probing to save the updated config."
        )

    # Model configuration
    seq_enc = cfg.get("sequence_encoder", "mamba2")
    if fold is not None:
        ckpt_path = experiment_dir / f"fold_{fold}" / "ckpt" / "classifier.pt"
    else:
        ckpt_path = experiment_dir / "ckpt" / "classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"classifier.pt not found at {ckpt_path}")

    raw = torch.load(ckpt_path, map_location=accelerator.device, weights_only=False)

    # Detect slice_pooling from weight keys
    has_abmil_keys = any(k.startswith("cobra.attn.") for k in raw.keys())
    if has_abmil_keys:
        slice_pooling = "abmil"
    else:
        slice_pooling = "cls"

    # FM fusion: detect router (cobra.fm_router.*)
    has_fm_router = any(k.startswith("cobra.fm_router.") for k in raw.keys())
    if has_fm_router:
        fm_pooling = "router"
    else:
        fm_pooling = "avg_pool"

    num_fms = cobra_cfg.get("num_fms", cfg.get("num_fms"))
    router_kwargs = None
    if has_fm_router:
        if num_fms is None:
            embed_key = "cobra.fm_router.fm_embed.weight"
            bias_key = "cobra.fm_router.fm_logit_bias.weight"
            if embed_key in raw:
                num_fms = int(raw[embed_key].shape[0])
            elif bias_key in raw:
                num_fms = int(raw[bias_key].shape[0])
            else:
                raise ValueError("Cannot infer num_fms for router from classifier.pt.")
        router_kwargs = dict(
            router_use_fm_embedding="cobra.fm_router.fm_embed.weight" in raw,
            router_use_fm_logit_bias="cobra.fm_router.fm_logit_bias.weight" in raw,
            router_use_fm_logit_scale="cobra.fm_router.fm_logit_scale.weight" in raw,
            router_mode=cfg.get("router_mode", "soft"),
            router_top_k=cfg.get("router_top_k", None),
            router_temperature=cfg.get("router_temperature", 1.0),
            router_learnable_temperature=cfg.get("router_learnable_temperature", False),
        )
    physical_pe = cobra_cfg.get("physical_pe", cfg.get("physical_pe", False))
    regional_tokens = int(cobra_cfg.get("regional_tokens", cfg.get("regional_tokens", 0)))
    pooling_target = cfg.get("pooling_target")
    raw_output_dim = cfg.get("raw_output_dim")

    has_within_slice = any(k.startswith("cobra.within_slice_agg.") for k in raw.keys())
    if has_within_slice:
        raise ValueError(
            "classifier.pt contains removed within_slice_agg weights. "
            "Re-run the experiment with global-only CLS or flattened regional-token features."
        )

    # Per-FM projection adapters: rebuild FM-keyed adapters when present (cobra.* prefix).
    per_fm_meta = {**cfg, **cobra_cfg}
    per_fm_adapter_mode, fm_input_dims = _resolve_per_fm_adapter(per_fm_meta, raw, prefix="cobra.")

    logger.info(
        "Loading COBRA from experiment: "
        f"sequence_encoder={seq_enc}, slice_pooling={slice_pooling}, "
        f"fm_pooling={fm_pooling}, pooling_target={pooling_target}, "
        f"physical_pe={physical_pe}, regional_tokens={regional_tokens}, "
        f"per_fm_adapter_mode={per_fm_adapter_mode}"
    )

    model = _build_cobra(
        cobra_cfg,
        seq_enc,
        slice_pooling,
        fm_pooling,
        pooling_target=pooling_target,
        raw_output_dim=raw_output_dim,
        physical_pe=physical_pe,
        regional_tokens=regional_tokens,
        num_fms=num_fms,
        per_fm_adapter_mode=per_fm_adapter_mode,
        fm_input_dims=fm_input_dims,
        router_kwargs=router_kwargs,
    )
    logger.info(f"Inference pooling_target={model.pooling_target}")

    # Extract cobra.* weights from classifier state dict
    cobra_weights = {}
    for k, v in raw.items():
        if not k.startswith("cobra."):
            continue
        new_key = k[len("cobra."):]
        cobra_weights[new_key] = v
    if not cobra_weights:
        raise ValueError("No cobra.* keys found in classifier.pt")

    model.load_state_dict(cobra_weights, strict=False)
    logger.info("COBRA model loaded from experiment directory.")

    return model, cfg
