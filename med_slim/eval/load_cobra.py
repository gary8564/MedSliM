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
    pooling_target: str = "post_embed",
    raw_output_dim: Optional[int] = None,
) -> Cobra:
    """Construct a COBRA model in inference mode from model configuration."""
    embed_dim = model_config["embed_dim"]
    encoder_kwargs: Dict[str, Any] = {}

    if sequence_encoder == "mamba2":
        encoder_kwargs["d_state"] = model_config.get("mamba_d_state", 128)
    else:
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
    pooling_target: str = "post_embed",
    raw_output_dim: Optional[int] = None,
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
    - pooling_target (str): Which representation to pool at inference ('post_encoder', 'post_embed', 'raw').
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
    logger.info(f"Loading COBRA with sequence_encoder={sequence_encoder}, slice_pooling={slice_pooling}, fm_pooling={fm_pooling}, pooling_target={pooling_target}")
    
    model = _build_cobra(model_config, sequence_encoder, slice_pooling, fm_pooling, pooling_target, raw_output_dim)
    
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
            #TODO: remove this once all checkpoints are updated
            # Skip removed varlen_seq_enc keys from old checkpoints
            if "varlen_seq_enc" in new_key:
                continue
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
    has_attn_keys = any(k.startswith("cobra.attn.") for k in raw.keys())
    slice_pooling = "abmil" if has_attn_keys else "cls"

    # Attention pooling if cobra.fm_attn.* exists
    has_fm_attn = any(k.startswith("cobra.fm_attn.") for k in raw.keys())
    fm_pooling = "attention" if has_fm_attn else "avg_pool"

    logger.info(
        f"Loading COBRA from experiment: sequence_encoder={seq_enc}, slice_pooling={slice_pooling}, fm_pooling={fm_pooling}"
    )

    model = _build_cobra(cobra_cfg, seq_enc, slice_pooling, fm_pooling)

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
