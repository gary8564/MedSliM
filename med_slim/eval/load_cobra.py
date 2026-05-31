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


def _build_cobra(
    model_config: Dict,
    sequence_encoder: str,
    slice_pooling: str,
    fm_pooling: str,
    pooling_target: Optional[str] = None,
    raw_output_dim: Optional[int] = None,
    physical_pe: Optional[bool] = None,
    regional_tokens: Optional[int] = None,
    region_embedding: Optional[bool] = None,
) -> Cobra:
    """Construct a COBRA model in inference mode from model configuration."""
    embed_dim = model_config["embed_dim"]
    encoder_kwargs: Dict[str, Any] = {}
    if physical_pe is None:
        physical_pe = model_config.get("physical_pe", False)
    if regional_tokens is None:
        regional_tokens = model_config.get("regional_tokens", 0)
    if region_embedding is None:
        region_embedding = model_config.get("region_embedding", False)

    if sequence_encoder == "mamba2":
        encoder_kwargs["d_state"] = model_config.get("mamba_d_state", 128)
    elif sequence_encoder == "transformer":
        encoder_kwargs["rotary_positional_encoding"] = model_config["transformer_rotary_positional_encoding"]
        encoder_kwargs["norm_first"] = model_config["transformer_norm_first"]
        encoder_kwargs["dim_feedforward"] = model_config.get(
            "transformer_dim_feedforward",
            4 * embed_dim,
        )

    if slice_pooling == "abmil" or fm_pooling == "attention":
        encoder_kwargs["att_dim"] = model_config["attn_dim"]

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
        region_embedding=region_embedding,
        **encoder_kwargs,
    )


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
     If None, Cobra resolves a safe default from mode and regional_tokens.
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
    regional_tokens = state_dict.get("regional_tokens", model_config.get("regional_tokens", 0))
    region_embedding = state_dict.get("region_embedding", model_config.get("region_embedding", False))
    logger.info(f"Loading COBRA with sequence_encoder={sequence_encoder}, slice_pooling={slice_pooling}, fm_pooling={fm_pooling}, pooling_target={pooling_target}, physical_pe={physical_pe}, regional_tokens={regional_tokens}")
    
    model = _build_cobra(model_config, sequence_encoder, slice_pooling, fm_pooling, pooling_target, raw_output_dim, physical_pe, regional_tokens, region_embedding)
    logger.info(f"Inference pooling_target={model.pooling_target}")
    
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
    has_cross_attn_keys = any(k.startswith("cobra.cross_attn_pool.") for k in raw.keys())
    if has_abmil_keys:
        slice_pooling = "abmil"
    elif has_cross_attn_keys:
        slice_pooling = "cross_attention"
    else:
        slice_pooling = "cls"

    # Attention pooling if cobra.fm_attn.* exists
    has_fm_attn = any(k.startswith("cobra.fm_attn.") for k in raw.keys())
    fm_pooling = "attention" if has_fm_attn else "avg_pool"
    physical_pe = cobra_cfg.get("physical_pe", cfg.get("physical_pe", False))
    pooling_target = cfg.get("pooling_target")
    raw_output_dim = cfg.get("raw_output_dim")

    # Tiled multi-crop CLS: detect within-slice aggregator from weight keys.
    has_within_slice = any(k.startswith("cobra.within_slice_agg.") for k in raw.keys())
    region_embed_key = "cobra.within_slice_agg.region_embed.weight"
    has_region_embed = region_embed_key in raw
    region_embedding = has_region_embed

    regional_tokens = int(cobra_cfg.get("regional_tokens", cfg.get("regional_tokens", 0)))
    if has_within_slice:
        if regional_tokens == 0:
            if has_region_embed:
                regional_tokens = int(raw[region_embed_key].shape[0]) - 1
                logger.warning(
                    "regional_tokens is specified as global-only setting, " 
                    "but the model checkpoint contains within_slice_agg weights."
                    "Inferred regional_tokens=%s from within_slice_agg.region_embed weights",
                    regional_tokens,
                )
            else:
                raise ValueError(
                    "classifier.pt contains within_slice_agg weights, but regional_tokens "
                    "is missing from cobra_config. Re-run linear probing after setting "
                    "regional_tokens in the saved experiment config."
                )
        elif has_region_embed:
            inferred = int(raw[region_embed_key].shape[0]) - 1
            if regional_tokens != inferred:
                raise ValueError(
                    f"regional_tokens={regional_tokens} in cobra_config conflicts with "
                    f"region_embed weights implying regional_tokens={inferred}."
                )
    elif regional_tokens > 0:
        raise ValueError(
            f"cobra_config specifies regional_tokens={regional_tokens}, but classifier.pt "
            "has no within_slice_agg weights."
        )

    logger.info(
        "Loading COBRA from experiment: "
        f"sequence_encoder={seq_enc}, slice_pooling={slice_pooling}, "
        f"fm_pooling={fm_pooling}, pooling_target={pooling_target}, "
        f"physical_pe={physical_pe}, regional_tokens={regional_tokens}"
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
        region_embedding=region_embedding,
    )
    logger.info(f"Inference pooling_target={model.pooling_target}")

    # Extract cobra.* weights from classifier state dict
    cobra_weights = {
        k[len("cobra."):]: v
        for k, v in raw.items()
        if k.startswith("cobra.")
    }
    if not cobra_weights:
        raise ValueError("No cobra.* keys found in classifier.pt")

    model.load_state_dict(cobra_weights, strict=False)
    logger.info("COBRA model loaded from experiment directory.")

    return model, cfg
