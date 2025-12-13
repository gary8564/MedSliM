import yaml
from pathlib import Path
from typing import Dict, Any

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model configurations from the YAML file.
    
    Args:
        model_name: Name of the pretrained model
        
    Returns:
        Dict containing model-specific configuration
    """
    current_dir = Path(__file__).parent
    configs_dir = current_dir.parent.parent / "configs"
    config_path = configs_dir / "pretrain.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            configs = yaml.safe_load(file)
        model_configs = configs["model"]["slice_encoder_models"]
        for model_config in model_configs:
            if model_config["name"] == model_name:
                return model_config.copy()
        raise ValueError(f"Model configuration not found: {model_name}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing model configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading model configuration file: {e}")