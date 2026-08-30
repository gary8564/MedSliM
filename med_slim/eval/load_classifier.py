"""
Load a shipped downstream classifier (COBRA + MLP head) from a linear-probing experiment.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import Accelerator

from med_slim.eval.linear_classifier import SingleViewClassifier
from med_slim.eval.load_cobra import load_cobra_from_experiment
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


def _infer_num_classes(classifier_weights: Dict[str, Any]) -> int:
    """
    Recover ``num_classes`` from the saved ClassifierHead output layer.

    ``ClassifierHead`` collapses binary tasks to a single logit, so an output
    dimension of 1 means ``num_classes=2``; otherwise the output dimension is
    the class count.
    """
    linear_layers: Dict[int, int] = {}
    for key, tensor in classifier_weights.items():
        parts = key.split(".")
        if len(parts) == 3 and parts[0] == "classifier" and parts[2] == "weight" and tensor.dim() == 2:
            linear_layers[int(parts[1])] = int(tensor.shape[0])
    if not linear_layers:
        raise ValueError(
            "Cannot infer num_classes: no Linear weight found in the classifier head."
        )
    out_features = linear_layers[max(linear_layers)]
    return 2 if out_features == 1 else out_features


def load_classifier_from_experiment(
    experiment_dir: str,
    accelerator: Accelerator,
    fold: Optional[int] = None,
) -> Tuple[SingleViewClassifier, Dict[str, Any]]:
    """
    Load the full downstream classifier (COBRA + MLP head) from a linear-probing experiment.

    Reuses ``load_cobra_from_experiment`` for the encoder and ``config.yml``, then
    rebuilds ``SingleViewClassifier`` and loads the full ``classifier.pt``
    (``cobra.*`` + ``classifier.*``).

    Args:
        experiment_dir: Linear-probing experiment directory holding ``config.yml``.
        accelerator: HuggingFace Accelerator (supplies the map location).
        fold: Load ``fold_{fold}/ckpt/classifier.pt`` instead of ``ckpt/classifier.pt``.

    Returns:
        (model, cfg) with the model in eval mode (dropout off) on the device it was
        built on; move it to the accelerator device at the call site.
    """
    cobra_model, cfg = load_cobra_from_experiment(experiment_dir, accelerator, fold=fold)

    if fold is not None:
        ckpt_path = Path(experiment_dir) / f"fold_{fold}" / "ckpt" / "classifier.pt"
    else:
        ckpt_path = Path(experiment_dir) / "ckpt" / "classifier.pt"
    raw = torch.load(ckpt_path, map_location=accelerator.device, weights_only=False)

    classifier_weights = {
        k[len("classifier."):]: v for k, v in raw.items() if k.startswith("classifier.")
    }
    if not classifier_weights:
        raise ValueError(
            f"No classifier.* keys found in {ckpt_path}. This checkpoint has no "
            "MLP head, so logit-based explanations are not possible."
        )
    num_classes = _infer_num_classes(classifier_weights)

    hyperparams = cfg.get("hyperparams", {})
    linear_hyperparams = hyperparams.get("linear", hyperparams)

    logger.info(
        "Loading downstream classifier from experiment: "
        f"num_classes={num_classes}, input_dim={cobra_model.output_dim}, "
        f"hidden_dim={linear_hyperparams.get('hidden_dim', 512)}, "
        f"eval_fm_ids={cfg.get('eval_fm_ids')}"
    )

    model = SingleViewClassifier(
        cobra_model=cobra_model,
        input_dim=cobra_model.output_dim,
        num_classes=num_classes,
        classifier_hidden_dim=linear_hyperparams.get("hidden_dim", 512),
        classifier_dropout=linear_hyperparams.get("dropout", 0.5),
        freeze_cobra=cfg.get("freeze_cobra", True),
        trainable_layers=cfg.get("trainable_layers"),
        fm_ids=cfg.get("eval_fm_ids"),
    )

    # fm_ids is a non-persistent buffer and the pretrain-only proj layer is absent
    # in inference mode, so only unexpected keys are a real error.
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in {ckpt_path}: {sorted(unexpected)}")
    head_missing = [k for k in missing if k.startswith("classifier.")]
    if head_missing:
        raise ValueError(f"Missing classifier head weights in {ckpt_path}: {head_missing}")

    model.eval()
    logger.info("Downstream classifier loaded from experiment directory.")
    return model, cfg
