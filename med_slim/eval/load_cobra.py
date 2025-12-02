# %%
import torch
import logging
import os
from accelerate import Accelerator
from typing import List

from med_slim.model.sequence_encoder.cobra import Cobra
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

def load_pretrained_cobra(checkpoint_path: str, 
                          accelerator: Accelerator, 
                          encoder_type: str = "momentum",
                          input_dims: List[int] = [512, 768, 1024, 1152, 1376]) -> Cobra:
    """
    Load the COBRA model from a pretrained checkpoint.

    Parameters:
    - checkpoint_path (str): Path to the model checkpoint file.
    - accelerator (Accelerator): HuggingFace Accelerator.
    - encoder_type (str): Choose between "base" and "momentum" encoder for downstream tasks. Default is "momentum".
    - input_dims (List[int]): List of input feature dimensions for the embedding module in pretrained model. Default is [512, 768, 1024, 1152, 1376].

    Returns:
    - Cobra: The loaded COBRA model.
    
    Raises:
    - FileNotFoundError: If the checkpoint file is not found.
    - ValueError: If the checkpoint format is invalid.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    state_dict = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)
    model = Cobra(input_dims=input_dims, mode="inference")
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
