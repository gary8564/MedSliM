import torch
import logging
import os
from accelerate import Accelerator
from typing import List, Dict

from med_slim.model.sequence_encoder.cobra import Cobra
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

def load_pretrained_cobra(checkpoint_path: str, 
                          accelerator: Accelerator, 
                          model_config: Dict,
                          encoder_type: str = "momentum") -> Cobra:
    """
    Load the COBRA model from a pretrained checkpoint.

    Parameters:
    - checkpoint_path (str): Path to the model checkpoint file.
    - accelerator (Accelerator): HuggingFace Accelerator.
    - model_config (Dict): Dictionary containing the model configuration.
    - encoder_type (str): Choose between "base" and "momentum" encoder for downstream tasks. Default is "momentum".

    Returns:
    - Cobra: The loaded COBRA model.
    
    Raises:
    - FileNotFoundError: If the checkpoint file is not found.
    - ValueError: If the checkpoint format is invalid.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    state_dict = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)
    model = Cobra(input_dims=model_config["input_dims"], 
                  embed_dim=model_config["embed_dim"],
                  contrast_dim=model_config["contrast_dim"],
                  num_heads=model_config["num_heads"],
                  layer=model_config["num_mamba_layers"],
                  dropout=model_config["dropout"],
                  att_dim=model_config["attn_dim"],
                  d_state=model_config["mamba_d_state"],
                  mode="inference")
    if "state_dict" in list(state_dict.keys()):
        chkpt = state_dict["state_dict"]
        cobra_weights = {
            k.split(f"{encoder_type}_encoder.")[-1]: v 
            for k, v in chkpt.items() 
            if f"{encoder_type}_encoder" in k and f"{encoder_type}_encoder.proj" not in k
        }
        if len(cobra_weights) == 0:
            raise ValueError(f"No {encoder_type} encoder weights found in checkpoint.")
    else:
        raise ValueError(f"`state_dict` key not found in saved model checkpoint {checkpoint_path}.")
    # strict=False: proj layer exists in pretrained model but excluded from checkpoint (not used in inference mode)
    model.load_state_dict(cobra_weights, strict=False)
    logger.info(f"{encoder_type.capitalize()} COBRA model loaded successfully.")
    return model
