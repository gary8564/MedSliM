import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

def load_eval_datasets_config() -> Dict[str, Any]:
    """Load the full ``eval_datasets.yaml`` config."""
    current_dir = Path(__file__).parent
    configs_dir = current_dir.parent / "configs"
    config_path = configs_dir / "eval_datasets.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"eval_datasets.yaml not found at {config_path}"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_metadata(dataset_name: str) -> Dict[str, Any]:
    """
    Return the metadata for specific dataset from `eval_datasets.yaml`.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dict containing dataset metadata
        
    Raises:
        KeyError: If the dataset is not registered
    """
    cfg = load_eval_datasets_config()
    datasets = cfg.get("datasets", {})
    if dataset_name not in datasets:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in eval_datasets.yaml. "
            f"Available: {list(datasets.keys())}"
        )
    return datasets[dataset_name]


def _resolve_annotation_path(
    annotations_dir: str,
    task: str,
    split: str,
) -> Optional[str]:
    """Resolve a single split name to its annotation CSV path, or None if absent."""
    base_dir = Path(annotations_dir)
    annot_path = base_dir / f"{split}.csv"
    if task == "binary" and not os.path.exists(annot_path):
        annot_path = base_dir / f"{split}_binary.csv"
    elif task == "multiclass" and not os.path.exists(annot_path):
        annot_path = base_dir / f"{split}_multiclass.csv"
    if not os.path.exists(annot_path):
        return None
    return str(annot_path)


def get_annotation_paths_by_split(
    annotations_dir: str,
    task: str,
    splits: List[str],
    optional_splits: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Resolve annotation CSV paths per split from a dataset annotations directory.

    For multiclass tasks, prefers ``{split}_multiclass.csv`` over ``{split}.csv``.

    Args:
        annotations_dir: Root directory containing annotation CSVs
        task: Classification task type (binary, multiclass, multilabel)
        splits: Required splits, e.g. ["train", "test"]. Missing required splits raise FileNotFoundError.
        optional_splits: Splits to include only if present on disk (e.g. ["val"]).
                         Missing optional splits are silently skipped instead of raising FileNotFoundError if they do not exist.

    Returns:
        Dict mapping split name to annotated csv path.
    """
    if not Path(annotations_dir).exists():
        raise FileNotFoundError(f"Annotations directory does not exist: {annotations_dir}")

    annot_paths: Dict[str, str] = {}
    for split in splits:
        resolved = _resolve_annotation_path(annotations_dir, task, split)
        if resolved is None:
            raise FileNotFoundError(
                f"Could not resolve annotations for required split '{split}' "
                f"in {annotations_dir}."
            )
        annot_paths[split] = resolved

    for split in optional_splits or []:
        resolved = _resolve_annotation_path(annotations_dir, task, split)
        if resolved is not None:
            annot_paths[split] = resolved

    return annot_paths

def format_label_name(raw_name: str) -> str:
    return raw_name.strip().replace("_", " ").title()


def format_display_name(
    raw_name: str,
    display_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Format a raw label column name to its human-readable display name.
    """
    key = raw_name.strip()
    if display_map:
        if key in display_map:
            return display_map[key]
        if key.lower() in display_map:
            return display_map[key.lower()]
    return format_label_name(key)


def build_binary_label_names(
    target_label: str,
    display_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Build a two-element list `[negative_name, positive_name]` for binary tasks.
    
    Args:
        target_label: Target label column name
        display_map: Optional display map
        
    Returns:
        List containing negative and positive label names
    """
    positive = format_display_name(target_label, display_map)
    return ["Normal", positive]


def build_multilabel_display_names(
    target_labels: List[str],
    display_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Build a list of display names in the same order as `target_labels`.
    
    Args:
        target_labels: List of target label column names
        display_map: Optional display map
        
    Returns:
        List containing display names
    """
    return [format_display_name(t, display_map) for t in target_labels]


def build_multiclass_label_names(
    target_label: str,
    multiclass_maps: Optional[Dict[str, Dict[int, str]]] = None,
) -> List[str]:
    """
    Build a dense list of class names for a multiclass column.

    Uses the `multiclass_label_maps` entry from `eval_datasets.yaml` when
    available, otherwise returns generic `Class 0`, `Class 1`, ... names.
    
    Args:
        target_label: Target label column name
        multiclass_maps: Optional multiclass label maps
        
    Returns:
        List containing class names
    """
    if multiclass_maps and target_label in multiclass_maps:
        label_map = multiclass_maps[target_label]
        max_idx = max(label_map.keys())
        names = [f"Class {i}" for i in range(max_idx + 1)]
        for idx, name in label_map.items():
            names[idx] = name
        return names
    return []
