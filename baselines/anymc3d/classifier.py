"""
Train an AnyMC3D 3D head on cached frozen 2D slice features.

Frozen FM + task-query pooling + linear classifier (Eq. 5 in arXiv:2512.12887).
There is no LoRA: the 2D backbone stays in the feature cache. See ``baselines/anymc3d/README.md``.

    python -m baselines.anymc3d.classifier --config baselines/anymc3d/configs/kneeMRI.yml --n-folds 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
import yaml
from accelerate import Accelerator
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup

from baselines.anymc3d.pooling import TaskQueryPooling, lengths_to_mask, slice_embeddings
from med_slim.data.feat_dataset import FeatClassificationDataset, linear_classifier_collate_fn
from med_slim.logging.setup import init_logging
from med_slim.utils.callbacks.early_stopping import EarlyStopping
from med_slim.utils.label_metadata import (
    build_multiclass_label_names,
    build_run_label_tag,
    get_annotation_paths_by_split,
    get_dataset_metadata,
)
from med_slim.utils.metrics.linear import (
    compute_class_weights_for_weighted_loss,
    get_eval_metrics,
    get_loss_criterion,
    get_num_classes,
)
from med_slim.utils.viz.linear import compute_and_visualize_metrics, compute_youden_thresholds

init_logging()
logger = logging.getLogger(__name__)

CURR_TIME = datetime.now().strftime("%Y-%m-%d-%H:%M")
JOB_ID = os.environ.get("SLURM_JOB_ID", str(os.getpid()))


def resolve_model_name(model_name) -> str:
    """AnyMC3D is a single-FM recipe; reject multi-FM lists."""
    if isinstance(model_name, (list, tuple)):
        names = [str(name) for name in model_name]
        if len(names) != 1:
            raise ValueError(
                "AnyMC3D uses one frozen 2D FM. Set model_name to a single name "
                f"(e.g. 'mri-core'), got {names}."
            )
        return names[0]
    if not model_name:
        raise ValueError("model_name is required (e.g. 'mri-core').")
    return str(model_name)


class AnyMC3DClassifier(nn.Module):
    """Slice embeddings → task-query pooling → linear/MLP head."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        query_init_std: float = 0.02,
        classifier: str = "linear",
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.pool = TaskQueryPooling(self.embed_dim, query_init_std=query_init_std)

        output_dim = 1 if num_classes == 2 else num_classes
        if classifier == "linear":
            self.classifier = nn.Linear(self.embed_dim, output_dim)
        elif classifier == "mlp":
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, classifier_hidden_dim),
                nn.SiLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(classifier_hidden_dim, output_dim),
            )
        else:
            raise ValueError(
                f"Unknown classifier '{classifier}'. Choose 'linear' (AnyMC3D) or 'mlp'."
            )

    def forward(
        self,
        features,
        seq_lengths: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        hidden = slice_embeddings(features)
        mask = lengths_to_mask(seq_lengths, hidden.size(1), device=hidden.device)
        if return_attention:
            pooled, attn = self.pool(hidden, mask=mask, return_attention=True)
        else:
            pooled = self.pool(hidden, mask=mask)
            attn = None

        out = {"logits": self.classifier(pooled), "pooled": pooled}
        if return_attention:
            out["attn_weights"] = attn
        return out


def _model_inputs(batch: Dict) -> Dict:
    return {"features": batch["features"], "seq_lengths": batch["seq_lengths"]}


def _compute_loss(logits, labels, task, criterion):
    if task == "binary":
        return criterion(logits.squeeze(-1), labels.float())
    if task == "multilabel":
        return criterion(logits, labels.float())
    return criterion(logits, labels)


def _compute_metrics(logits, labels, task, num_classes, device):
    metrics = get_eval_metrics(task, num_classes, device)
    logits, labels = logits.to(device), labels.to(device)
    if task == "binary":
        labels = labels.long()
        probs = torch.sigmoid(logits.squeeze(-1) if logits.ndim > 1 else logits)
    elif task == "multilabel":
        labels = labels.long()
        probs = torch.sigmoid(logits)
    else:
        probs = torch.softmax(logits, dim=-1)
    for metric in metrics.values():
        metric.update(probs, labels)
    return {key: value.compute().item() for key, value in metrics.items()}


def _build_predictions_dataframe(
    probs: np.ndarray,
    true_labels: np.ndarray,
    sample_ids: List[str],
    task: str,
    target_labels: List[str],
    thresholds: Optional[np.ndarray | float] = None,
) -> pd.DataFrame:
    """Format predictions like MedSliM linear evaluation output."""
    if task == "binary":
        probs = probs.squeeze(-1) if probs.ndim > 1 else probs
        threshold = 0.5 if thresholds is None else float(thresholds)
        pred_labels = (probs >= threshold).astype(int)
        return pd.DataFrame({
            "exam_id": sample_ids,
            "true_labels": true_labels.tolist(),
            "pred_labels": pred_labels.tolist(),
            "pred_probs": probs.tolist(),
        })

    if task == "multilabel":
        threshold_arr = (
            np.full(probs.shape[1], 0.5)
            if thresholds is None
            else np.asarray(thresholds, dtype=float)
        )
        pred_labels = (probs >= threshold_arr).astype(int)
        df_results = pd.DataFrame({"exam_id": sample_ids})
        df_results["true_labels"] = [list(map(int, row)) for row in true_labels]
        df_results["pred_labels"] = [
            [target_labels[i] for i, pred in enumerate(row) if pred == 1]
            for row in pred_labels
        ]
        df_results["pred_probs"] = [list(row) for row in probs]
        return df_results

    pred_labels = probs.argmax(axis=-1)
    return pd.DataFrame({
        "exam_id": sample_ids,
        "true_labels": true_labels.tolist(),
        "pred_labels": pred_labels.tolist(),
        "pred_probs": [list(row) for row in probs],
    })


def train_one_epoch(model, loader, optimizer, scheduler, task, criterion, num_classes, accelerator):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for batch in loader:
        optimizer.zero_grad()
        labels = batch["labels"]
        with accelerator.autocast():
            out = model(**_model_inputs(batch))
            loss = _compute_loss(out["logits"], labels, task, criterion)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        all_logits.append(out["logits"].detach())
        all_labels.append(labels.detach())
    avg_loss = total_loss / len(loader)
    logits_cat = accelerator.gather_for_metrics(torch.cat(all_logits))
    labels_cat = accelerator.gather_for_metrics(torch.cat(all_labels))
    metrics = _compute_metrics(logits_cat, labels_cat, task, num_classes, accelerator.device)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, task, criterion, num_classes, accelerator):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels, all_ids = [], [], []
    for batch in loader:
        labels = batch["labels"]
        with accelerator.autocast():
            out = model(**_model_inputs(batch))
            loss = _compute_loss(out["logits"], labels, task, criterion)
        total_loss += loss.item()
        all_logits.append(out["logits"].detach())
        all_labels.append(labels.detach())
        all_ids.extend(batch["sample_ids"])
    avg_loss = total_loss / max(len(loader), 1)
    logits_cat = accelerator.gather_for_metrics(torch.cat(all_logits))
    labels_cat = accelerator.gather_for_metrics(torch.cat(all_labels))
    metrics = _compute_metrics(logits_cat, labels_cat, task, num_classes, accelerator.device)
    return avg_loss, metrics, logits_cat, labels_cat, all_ids


def _logits_to_probs(logits: torch.Tensor, task: str) -> np.ndarray:
    if task == "binary":
        return torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    if task == "multilabel":
        return torch.sigmoid(logits).cpu().numpy()
    return torch.softmax(logits, dim=-1).cpu().numpy()


def _build_model(model_cfg: Dict, num_classes: int) -> AnyMC3DClassifier:
    return AnyMC3DClassifier(
        embed_dim=int(model_cfg.get("embed_dim", 768)),
        num_classes=num_classes,
        query_init_std=float(model_cfg.get("query_init_std", 0.02)),
        classifier=model_cfg.get("classifier", "linear"),
        classifier_hidden_dim=int(model_cfg.get("classifier_hidden_dim", 512)),
        classifier_dropout=float(model_cfg.get("classifier_dropout", 0.5)),
    )


def _build_optimizer(parameters, hp: Dict) -> torch.optim.Optimizer:
    """AdamW is the AnyMC3D query/head recipe; SGD is optional for protocol matching."""
    name = str(hp.get("optimizer", "adamw")).lower()
    lr = float(hp.get("lr", 1e-3))
    weight_decay = float(hp.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(hp.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer '{name}'. Choose 'adamw' or 'sgd'.")


def _build_data_loaders(train_ds, val_ds, test_ds, hp: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = hp.get("batch_size", 8)
    num_workers = hp.get("num_workers", 4)
    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=linear_classifier_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def _collect_train_labels(dataset: FeatClassificationDataset) -> np.ndarray:
    return np.array([dataset._get_label(sid).numpy() for sid in dataset.sample_ids])


def _derive_fold_seed(seed: int) -> int:
    """Match MedSliM linear probing random seed."""
    return int(np.random.SeedSequence(int(seed)).generate_state(1)[0])


def _fold_splits(labels: np.ndarray, n_folds: int, task: str, seed: int):
    """Stratified k-fold on train, same splitters as MedSliM linear probing."""
    indices = np.arange(len(labels))
    if task == "multilabel":
        kfold = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(kfold.split(X=indices, y=labels))
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kfold.split(X=indices, y=labels))


def _train_val_indices(labels: np.ndarray, val_split_ratio: float, task: str, seed: int):
    """Single train/val split, matching MedSliM LP when ``n_folds=1``."""
    indices = np.arange(len(labels))
    if task == "multilabel":
        n_splits = max(2, int(1.0 / val_split_ratio))
        kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = list(kfold.split(X=indices, y=labels))[0]
        return train_idx, val_idx
    return train_test_split(
        indices,
        test_size=val_split_ratio,
        stratify=labels,
        shuffle=True,
        random_state=seed,
    )


def _build_feat_dataset(
    cfg: Dict,
    split: str,
    annotations_path: str,
    task: str,
    target_labels: List[str],
) -> FeatClassificationDataset:
    model_name = resolve_model_name(cfg.get("model_name", "mri-core"))
    hp = cfg.get("hyperparams", {})
    return FeatClassificationDataset(
        feat_dir=cfg["feat_dir"],
        slice_encoder_models=[model_name],
        view_plane=cfg.get("plane", "sagittal"),
        split=split,
        annotations_path=annotations_path,
        task=task,
        target_columns=target_labels,
        cache_in_memory=bool(hp.get("cache_in_memory", True)),
    )


def run_one_split(
    *,
    train_ds,
    val_ds,
    test_ds,
    cfg: Dict,
    task: str,
    target_labels: List[str],
    class_names: Optional[List[str]],
    num_classes: int,
    train_annots: str,
    accelerator: Accelerator,
    output_dir: str,
    log_prefix: str = "",
    log_params: bool = True,
) -> Dict:
    """Train one head and evaluate on the test set."""
    hp = cfg["hyperparams"]
    model_cfg = cfg.get("model", {})
    train_loader, val_loader, test_loader = _build_data_loaders(train_ds, val_ds, test_ds, hp)
    model = _build_model(model_cfg, num_classes)

    if log_params and accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Trainable parameters: %s (frozen 2D FM lives in the feature cache)",
            f"{trainable:,}",
        )

    optimizer = _build_optimizer([p for p in model.parameters() if p.requires_grad], hp)
    max_epochs = hp.get("max_epochs", 100)
    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(total_steps * hp.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    class_weights = (
        compute_class_weights_for_weighted_loss(train_annots, target_labels, task, accelerator.device)
        if hp.get("weighted_loss", False) else None
    )
    criterion = get_loss_criterion(task, class_weights)

    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler,
    )

    patience = hp.get("patience", 10)
    ckpt_dir = os.path.join(output_dir, "ckpt")
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
    early_stopping = EarlyStopping(
        patience=patience, mode="max",
        ckpt_path=os.path.join(ckpt_dir, "best.pt"),
        accelerator=accelerator,
    ) if patience > 0 else None

    best_val_auroc = -1.0
    for epoch in range(max_epochs):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, task, criterion, num_classes, accelerator,
        )
        val_loss, val_metrics, *_ = evaluate(
            model, val_loader, task, criterion, num_classes, accelerator,
        )
        val_auroc = val_metrics.get("auroc", 0.0)

        if accelerator.is_main_process:
            log_dict = {
                "lr": scheduler.get_last_lr()[0],
                f"{log_prefix}train/loss": train_loss,
                f"{log_prefix}val/loss": val_loss,
                "epoch": epoch + 1,
            }
            for key, value in train_metrics.items():
                log_dict[f"{log_prefix}train/{key}"] = value
            for key, value in val_metrics.items():
                log_dict[f"{log_prefix}val/{key}"] = value
            wandb.log(log_dict)
            if epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_auroc={val_auroc:.4f}"
                )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc

        if early_stopping is not None:
            should_stop, best_score = early_stopping.step(val_auroc, model, optimizer, scheduler, epoch)
            if should_stop:
                if accelerator.is_main_process:
                    logger.info(f"Early stopping at epoch {epoch+1}, best AUROC={best_score:.4f}")
                unwrapped = accelerator.unwrap_model(model)
                early_stopping.load_best_model(unwrapped)
                break

    _, _, val_logits, val_labels, _ = evaluate(
        model, val_loader, task, criterion, num_classes, accelerator,
    )
    val_thresholds = None
    if accelerator.is_main_process:
        val_thresholds = compute_youden_thresholds(
            y_true=val_labels.cpu().numpy(),
            y_pred_prob=_logits_to_probs(val_logits, task),
            task=task,
        )

    if accelerator.is_main_process:
        logger.info("Evaluating on test set ...")
    _, test_metrics, test_logits, test_labels, test_ids = evaluate(
        model, test_loader, task, criterion, num_classes, accelerator,
    )

    result = {
        "auroc": float(test_metrics.get("auroc", 0.0)),
        "auprc": float(test_metrics.get("auprc", 0.0)),
        "best_val_auroc": float(best_val_auroc),
        "test_metrics": test_metrics,
    }

    if accelerator.is_main_process:
        logger.info(f"Test metrics: {test_metrics}")
        wandb.log({f"{log_prefix}test/{k}": v for k, v in test_metrics.items()})

        probs = _logits_to_probs(test_logits, task)
        true_np = test_labels.cpu().numpy()
        viz_labels = class_names if class_names else target_labels
        fig_dir = os.path.join(output_dir, "fig")
        compute_and_visualize_metrics(
            y_true=true_np, y_pred_prob=probs, task=task,
            class_labels=viz_labels, output_dir=fig_dir, thresholds=val_thresholds,
        )
        logger.info(f"Plots saved to {fig_dir}")

        results = {
            "test_metrics": {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in test_metrics.items()
            },
            "best_val_auroc": float(best_val_auroc),
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        table_dir = os.path.join(output_dir, "table")
        os.makedirs(table_dir, exist_ok=True)
        predictions = _build_predictions_dataframe(
            probs=probs,
            true_labels=true_np,
            sample_ids=test_ids,
            task=task,
            target_labels=target_labels,
            thresholds=val_thresholds,
        )
        predictions.to_csv(os.path.join(table_dir, "predictions.csv"), index=False)
        logger.info(f"Predictions saved to {os.path.join(table_dir, 'predictions.csv')}")

    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    accelerator.wait_for_everyone()
    return result


def main(args):
    accelerator = Accelerator()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.feat_dir:
        cfg["feat_dir"] = args.feat_dir
    if args.annotations_dir:
        cfg["annotations_dir"] = args.annotations_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.model_name:
        cfg["model_name"] = args.model_name
    if args.plane:
        cfg["plane"] = args.plane
    if args.weighted_loss:
        cfg.setdefault("hyperparams", {})["weighted_loss"] = True

    n_folds = args.n_folds
    fold_seed = _derive_fold_seed(args.seed)
    cfg["n_folds"] = n_folds
    cfg["seed"] = args.seed
    cfg["fold_seed"] = fold_seed
    cfg["model_name"] = resolve_model_name(cfg.get("model_name", "mri-core"))
    cfg.setdefault("model", {})

    if not cfg.get("feat_dir"):
        raise ValueError(
            "feat_dir is required: MedSliM slice-feature cache root "
            "({feat_dir}/{model_name}/{split}/{plane}/*.safetensors)."
        )

    dataset_name = cfg["dataset_name"]
    ds_meta = get_dataset_metadata(dataset_name)
    task = ds_meta["task"]
    target_labels = ds_meta["target_labels"]
    class_names = None
    if task == "multiclass" and "multiclass_label_maps" in ds_meta:
        class_names = build_multiclass_label_names(target_labels[0], ds_meta["multiclass_label_maps"])

    annotations_dir = cfg["annotations_dir"]
    annot_paths = get_annotation_paths_by_split(
        annotations_dir,
        task,
        splits=["train", "test"],
        optional_splits=["val"],
    )
    train_annots = annot_paths["train"]
    test_annots = annot_paths["test"]
    val_annots = annot_paths.get("val")

    if n_folds > 1 and val_annots:
        if accelerator.is_main_process:
            logger.warning(
                f"--n-folds={n_folds} ignored because a dedicated validation set is "
                "available (same rule as linear_classifier.py)."
            )
        n_folds = 1
        cfg["n_folds"] = 1

    plane = cfg.get("plane", "sagittal")
    hp = cfg["hyperparams"]
    model_cfg = cfg["model"]
    model_name = cfg["model_name"]

    if accelerator.is_main_process:
        logger.info(
            "AnyMC3D recipe: fm=%s, pooling=query, head=%s, embed_dim=%s",
            model_name,
            model_cfg.get("classifier", "linear"),
            model_cfg.get("embed_dim", 768),
        )
        logger.info(
            "Split protocol: n_folds=%s, seed=%s, fold_seed=%s (SeedSequence, same as LP --seed)",
            n_folds, args.seed, fold_seed,
        )

    labeled_train_ds = _build_feat_dataset(cfg, "train", train_annots, task, target_labels)
    test_ds = _build_feat_dataset(cfg, "test", test_annots, task, target_labels)
    num_classes = get_num_classes(task, target_labels, test_ds)

    run_label_tag = build_run_label_tag(task, target_labels, plane)
    output_dir = os.path.join(
        cfg.get("output_dir", "experiments/anymc3d_classifier"),
        f"{dataset_name}_{model_name}_{run_label_tag}_{CURR_TIME}_{JOB_ID}",
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yml"), "w") as f:
            yaml.dump(cfg, f)
        wandb.init(
            project="anymc3d-classifier",
            name=f"{dataset_name}_{model_name}_{run_label_tag}",
            config=cfg,
            dir=output_dir,
        )

    split_kwargs = dict(
        test_ds=test_ds,
        cfg=cfg,
        task=task,
        target_labels=target_labels,
        class_names=class_names,
        num_classes=num_classes,
        train_annots=train_annots,
        accelerator=accelerator,
    )

    if n_folds <= 1:
        if val_annots:
            val_ds = _build_feat_dataset(cfg, "val", val_annots, task, target_labels)
            train_ds = labeled_train_ds
        else:
            all_labels = _collect_train_labels(labeled_train_ds)
            train_idx, val_idx = _train_val_indices(
                all_labels, hp.get("val_split_ratio", 0.1), task, fold_seed
            )
            val_ds = Subset(labeled_train_ds, val_idx)
            train_ds = Subset(labeled_train_ds, train_idx)

        if accelerator.is_main_process:
            logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

        run_one_split(
            train_ds=train_ds, val_ds=val_ds, output_dir=output_dir, **split_kwargs,
        )
    else:
        all_labels = _collect_train_labels(labeled_train_ds)
        fold_splits = _fold_splits(all_labels, n_folds, task, seed=fold_seed)
        fold_aurocs, fold_auprcs = [], []

        if accelerator.is_main_process:
            logger.info("=" * 50)
            logger.info(f"{n_folds}-Fold Cross-Validation Evaluation")
            logger.info("=" * 50)

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            train_ds = Subset(labeled_train_ds, train_idx.tolist())
            val_ds = Subset(labeled_train_ds, val_idx.tolist())
            fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
            if accelerator.is_main_process:
                os.makedirs(fold_dir, exist_ok=True)
                logger.info("=" * 50)
                logger.info(f"Fold {fold_idx + 1}/{n_folds}")
                logger.info(f"Training: {len(train_ds)}, Validation: {len(val_ds)}, Test: {len(test_ds)}")
                logger.info("=" * 50)

            fold_result = run_one_split(
                train_ds=train_ds,
                val_ds=val_ds,
                output_dir=fold_dir,
                log_prefix=f"fold{fold_idx + 1}/",
                log_params=(fold_idx == 0),
                **split_kwargs,
            )
            if accelerator.is_main_process:
                fold_aurocs.append(fold_result["auroc"])
                fold_auprcs.append(fold_result["auprc"])
                logger.info(
                    f"Fold {fold_idx + 1}: AUROC={fold_result['auroc']:.4f}, "
                    f"AUPRC={fold_result['auprc']:.4f}"
                )
                wandb.log({
                    f"fold{fold_idx + 1}/test_auroc": fold_result["auroc"],
                    f"fold{fold_idx + 1}/test_auprc": fold_result["auprc"],
                })

        if accelerator.is_main_process and fold_aurocs:
            mean_auroc = float(np.mean(fold_aurocs))
            std_auroc = float(np.std(fold_aurocs))
            mean_auprc = float(np.mean(fold_auprcs))
            std_auprc = float(np.std(fold_auprcs))
            logger.info("=" * 50)
            logger.info(f"{n_folds}-Fold Cross-Validation Results")
            logger.info("=" * 50)
            for i, (auc_val, prc_val) in enumerate(zip(fold_aurocs, fold_auprcs)):
                logger.info(f"  Fold {i + 1}: AUROC={auc_val:.4f}, AUPRC={prc_val:.4f}")
            logger.info(f"  Mean AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}")
            logger.info(f"  Mean AUPRC: {mean_auprc:.4f} +/- {std_auprc:.4f}")

            cv_summary = {
                "n_folds": n_folds,
                "seed": args.seed,
                "fold_seed": fold_seed,
                "model_name": model_name,
                "pooling": "query",
                "fold_aurocs": fold_aurocs,
                "fold_auprcs": fold_auprcs,
                "mean_auroc": mean_auroc,
                "std_auroc": std_auroc,
                "mean_auprc": mean_auprc,
                "std_auprc": std_auprc,
            }
            table_dir = os.path.join(output_dir, "table")
            os.makedirs(table_dir, exist_ok=True)
            with open(os.path.join(table_dir, "cv_summary.json"), "w") as f:
                json.dump(cv_summary, f, indent=4)
            with open(os.path.join(output_dir, "results.json"), "w") as f:
                json.dump(cv_summary, f, indent=2)
            wandb.run.summary["cv_mean_auroc"] = mean_auroc
            wandb.run.summary["cv_std_auroc"] = std_auroc
            wandb.run.summary["cv_mean_auprc"] = mean_auprc
            wandb.run.summary["cv_std_auprc"] = std_auprc
            wandb.run.summary["test_auroc"] = mean_auroc
            wandb.run.summary["test_auprc"] = mean_auprc

    if accelerator.is_main_process:
        wandb.finish()
        logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AnyMC3D classifier on cached 2D FM features (task-query pooling + linear head).",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="YAML config (see baselines/anymc3d/configs/kneeMRI.yml).",
    )
    parser.add_argument(
        "--feat-dir", type=str, default=None,
        help="Override config feat_dir (MedSliM slice-feature cache root).",
    )
    parser.add_argument(
        "--annotations-dir", type=str, default=None,
        help="Override config annotations_dir.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override config output_dir.",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Frozen 2D FM whose cache to read (default: config / mri-core).",
    )
    parser.add_argument(
        "--plane", type=str, default=None,
        help="Override config plane (e.g. sagittal).",
    )
    parser.add_argument(
        "--weighted-loss", action="store_true",
        help="Class-weighted loss (same flag as MedSliM linear probing).",
    )
    parser.add_argument(
        "--n-folds", type=int, default=1,
        help="Stratified k-fold on official train, official test held out every "
             "fold. Ignored when a dedicated val split exists. (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed. Fold random_state is SeedSequence(seed).generate_state(1)[0], "
             "matching MedSliM linear probing.",
    )
    main(parser.parse_args())
