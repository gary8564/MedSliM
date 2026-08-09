"""
End-to-end Curia evaluation: frozen backbone → cross-attention pooling → classifier.

Runs directly on NIfTI volumes (no precomputed feature cache required).

Usage:
    python -m med_slim.eval.curia_classifier \
        --config med_slim/configs/curia_classifier.yml

Architecture:
    NIfTI volume → Curia backbone (frozen) → per-slice tokens
        → (optional spatial avg-pool) → cross-attention pooling (trainable)
        → classifier head (trainable) → logits
"""

import os
import argparse
import json
import yaml
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from datetime import datetime
from typing import Dict, List, Optional
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from med_slim.model.slice_encoder.curia import CuriaFeatureExtractor
from med_slim.model.attention_pooling.cross_attention import InterSliceAggregator
from med_slim.data.slice_dataset import SliceDataset
from med_slim.utils.preprocessing.transforms import get_adaptive_transform, get_transforms
from med_slim.utils.callbacks.early_stopping import EarlyStopping
from med_slim.utils.label_metadata import (
    get_dataset_metadata,
    get_annotation_paths_by_split,
    build_multiclass_label_names,
    build_run_label_tag,
)
from med_slim.utils.metrics.linear import (
    get_loss_criterion,
    get_eval_metrics,
    get_num_classes,
    compute_class_weights_for_weighted_loss,
)
from med_slim.utils.viz.linear import compute_and_visualize_metrics, compute_youden_thresholds
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

CURR_TIME = datetime.now().strftime("%Y-%m-%d-%H:%M")
JOB_ID = os.environ.get("SLURM_JOB_ID", str(os.getpid()))


# Model
class CuriaClassifier(nn.Module):
    """
    End-to-end model:  Curia backbone (frozen) → cross-attention pooling → MLP classifier.

    The backbone extracts per-slice CLS tokens, patch tokens, or both from a 3D volume.
    Optional sinusoidal slice positional embeddings mirror Curia's 3D feature
    extraction code by adding the same slice-index embedding to every token
    belonging to that slice before cross-attention pooling.
    Cross-attention pooling learns which slices are diagnostically relevant
    and produces a single volume-level embedding.  A lightweight MLP maps
    that embedding to class logits.
    """

    def __init__(
        self,
        model_repo: str = "raidium/curia",
        local_cache_dir: Optional[str] = None,
        token_mode: str = "cls",
        spatial_pool_kernel_size: Optional[int] = None,
        num_classes: int = 2,
        num_heads: int = 8,
        num_queries: int = 1,
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.5,
        pooling_dropout: float = 0.0,
        add_slice_positional_embedding: bool = True,
    ):
        super().__init__()
        self.token_mode = token_mode
        self.add_slice_positional_embedding = add_slice_positional_embedding

        # Frozen backbone
        self.backbone = CuriaFeatureExtractor(
            model_repo=model_repo,
            local_cache_dir=local_cache_dir,
            token_mode=token_mode,
            spatial_pool_kernel_size=spatial_pool_kernel_size,
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        embed_dim = self.backbone.embed_dim

        # Trainable aggregation
        self.cross_attn_pool = InterSliceAggregator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_queries=num_queries,
            dropout=pooling_dropout,
        )

        # Trainable classifier
        self.num_classes = num_classes
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, classifier_hidden_dim),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, output_dim),
        )

    @staticmethod
    def _slice_positional_embeddings(
        num_slices: int,
        embed_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sinusoidal positional embeddings over discrete slice indices."""
        positions = torch.arange(num_slices, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(num_slices, embed_dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        return pe

    def _add_slice_positional_embedding(
        self,
        h: torch.Tensor,
        num_slices: int,
    ) -> torch.Tensor:
        """
        Add the same slice-index positional embedding to every token from a slice.

        CuriaFeatureExtractor emits tokens grouped by slice:
            cls:       [slice_0_cls, slice_1_cls, ...]
            patch:     [slice_0_patch_*, slice_1_patch_*, ...]
            cls_patch: [slice_0_cls, slice_0_patch_*, slice_1_cls, ...]
        """
        if h.shape[1] % num_slices != 0:
            raise ValueError(
                f"Cannot add slice positional embeddings: token count {h.shape[1]} "
                f"is not divisible by num_slices={num_slices}."
            )

        tokens_per_slice = h.shape[1] // num_slices
        pe = self._slice_positional_embeddings(
            num_slices=num_slices,
            embed_dim=h.shape[-1],
            device=h.device,
            dtype=h.dtype,
        )
        pe = pe.repeat_interleave(tokens_per_slice, dim=0).unsqueeze(0)
        return h + pe

    @staticmethod
    def _expand_slice_mask_to_tokens(
        slice_mask: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        """
        Expand a [B, D] valid-slice mask to the token sequence emitted by Curia.
        Curia tokens are grouped by slice, so each slice mask value repeats for
        the number of tokens produced per slice.
        """
        num_slices = slice_mask.shape[1]
        if token_count % num_slices != 0:
            raise ValueError(
                f"Cannot expand slice mask: token count {token_count} "
                f"is not divisible by num_slices={num_slices}."
            )
        tokens_per_slice = token_count // num_slices
        return slice_mask.repeat_interleave(tokens_per_slice, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        slice_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Volume tensor [B, C, W, H, D], C=1.
            slice_mask: Boolean mask [B, D], True for real slices and False for padding.
            return_attention: Also return cross-attention weights.

        Returns:
            Dict with ``logits`` and optionally ``attn_weights``.
        """
        with torch.no_grad():
            h = self.backbone(x)  # [B, T, embed_dim]
        if self.add_slice_positional_embedding:
            h = self._add_slice_positional_embedding(h, num_slices=x.shape[-1])
        token_mask = None
        if slice_mask is not None:
            token_mask = self._expand_slice_mask_to_tokens(
                slice_mask=slice_mask.to(device=h.device, dtype=torch.bool),
                token_count=h.shape[1],
            )

        if return_attention:
            pooled, attn = self.cross_attn_pool(h, mask=token_mask, return_attention=True)
            logits = self.classifier(pooled)
            return {"logits": logits, "attn_weights": attn}

        pooled = self.cross_attn_pool(h, mask=token_mask)  # [B, embed_dim]
        logits = self.classifier(pooled)
        return {"logits": logits}


# Dataset wrapper: adds labels to SliceDataset
class LabeledSliceDataset(SliceDataset):
    """SliceDataset augmented with classification labels from a CSV."""

    def __init__(
        self,
        path_root: str,
        split: str,
        annotations_path: str,
        task: str,
        target_columns: List[str],
        transform=None,
        plane: str = "sagittal",
    ):
        super().__init__(path_root=path_root, split=split, transform=transform, plane=plane)
        self.task = task
        self.target_columns = target_columns

        self.df_labels = pd.read_csv(annotations_path, dtype={"ID": str})
        self.df_labels.set_index("ID", inplace=True)

        # Keep only samples that have both NIfTI files and annotations
        annotated_ids = set(self.df_labels.index)
        self.sample_ids = [sid for sid in self.sample_ids if sid in annotated_ids]
        if len(self.sample_ids) == 0:
            raise ValueError(
                f"No overlapping sample IDs between NIfTI files in "
                f"{path_root}/{split}/{plane} and annotations in {annotations_path}."
            )

    def _get_label(self, sample_id: str) -> torch.Tensor:
        if self.task == "multilabel":
            vals = self.df_labels.loc[sample_id, self.target_columns].values.astype(np.float32)
            return torch.tensor(vals, dtype=torch.float32)
        label = self.df_labels.loc[sample_id, self.target_columns[0]]
        return torch.tensor(label, dtype=torch.long)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        label = self._get_label(item["uid"])
        item["label"] = label
        return item


def labeled_collate_fn(batch):
    """Collate for LabeledSliceDataset: pad variable-depth volumes and labels."""
    uids = [s["uid"] for s in batch]
    tensors = [s["source"].tensor for s in batch]  # each (C, W, H, D)
    labels = torch.stack([s["label"] for s in batch])
    max_slices = max(t.shape[-1] for t in tensors)

    padded_tensors = []
    slice_masks = []
    for tensor in tensors:
        num_slices = tensor.shape[-1]
        padded = tensor.new_zeros(*tensor.shape[:-1], max_slices)
        padded[..., :num_slices] = tensor
        padded_tensors.append(padded)

        mask = torch.zeros(max_slices, dtype=torch.bool)
        mask[:num_slices] = True
        slice_masks.append(mask)

    x = torch.stack(padded_tensors, dim=0)  # (B, C, W, H, D_max)
    slice_mask = torch.stack(slice_masks, dim=0)  # (B, D_max)
    return {"uid": uids, "x": x, "labels": labels, "slice_mask": slice_mask}


# Training / evaluation helpers
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
    for m in metrics.values():
        m.update(probs, labels)
    return {k: v.compute().item() for k, v in metrics.items()}


def _build_predictions_dataframe(
    probs: np.ndarray,
    true_labels: np.ndarray,
    sample_ids: List[str],
    task: str,
    target_labels: List[str],
    thresholds: Optional[np.ndarray | float] = None,
) -> pd.DataFrame:
    """Format Curia classifier predictions like MedSliM linear evaluation output."""
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
        x = batch["x"]
        labels = batch["labels"]
        slice_mask = batch["slice_mask"]
        with accelerator.autocast():
            out = model(x, slice_mask=slice_mask)
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
        x = batch["x"]
        labels = batch["labels"]
        slice_mask = batch["slice_mask"]
        with accelerator.autocast():
            out = model(x, slice_mask=slice_mask)
            loss = _compute_loss(out["logits"], labels, task, criterion)
        total_loss += loss.item()
        all_logits.append(out["logits"].detach())
        all_labels.append(labels.detach())
        all_ids.extend(batch["uid"])
    avg_loss = total_loss / max(len(loader), 1)
    logits_cat = accelerator.gather_for_metrics(torch.cat(all_logits))
    labels_cat = accelerator.gather_for_metrics(torch.cat(all_labels))
    metrics = _compute_metrics(logits_cat, labels_cat, task, num_classes, accelerator.device)
    return avg_loss, metrics, logits_cat, labels_cat, all_ids


# Main
def main(args):
    accelerator = Accelerator()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve dataset metadata
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

    plane = cfg.get("plane", "sagittal")
    data_dir = cfg["data_dir"]
    hp = cfg["hyperparams"]

    # Transforms
    spatial_mode = cfg.get("spatial_mode", "adaptive")
    modality = cfg.get("modality", "mri")
    if spatial_mode == "adaptive":
        val_transform = get_adaptive_transform(
            model_name="curia",
            plane=plane,
            num_slices=cfg.get("num_slices"),
            crop_empty_slices=cfg.get("crop_empty_slices", False),
            modality=modality,
            to_tensor=False,
        )
    else:
        _, val_transform = get_transforms(
            model_name="curia",
            plane=plane,
            num_slices=cfg.get("num_slices"),
            spatial_mode=spatial_mode,
            crop_empty_slices=cfg.get("crop_empty_slices", False),
            modality=modality,
            to_tensor=False,
        )

    # Datasets
    labeled_train_ds = LabeledSliceDataset(
        path_root=data_dir, split="train", annotations_path=train_annots,
        task=task, target_columns=target_labels, transform=val_transform, plane=plane,
    )
    test_ds = LabeledSliceDataset(
        path_root=data_dir, split="test", annotations_path=test_annots,
        task=task, target_columns=target_labels, transform=val_transform, plane=plane,
    )

    # Train / val split
    if val_annots:
        val_ds = LabeledSliceDataset(
            path_root=data_dir, split="val", annotations_path=val_annots,
            task=task, target_columns=target_labels, transform=val_transform, plane=plane,
        )
        train_ds = labeled_train_ds
    else:
        all_labels = np.array([
            labeled_train_ds._get_label(sid).numpy() for sid in labeled_train_ds.sample_ids
        ])
        train_idx, val_idx = train_test_split(
            np.arange(len(labeled_train_ds)),
            test_size=hp.get("val_split_ratio", 0.1),
            stratify=all_labels,
            shuffle=True,
        )
        val_ds = Subset(labeled_train_ds, val_idx)
        train_ds = Subset(labeled_train_ds, train_idx)

    if accelerator.is_main_process:
        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    batch_size = hp.get("batch_size", 4)
    num_workers = hp.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=labeled_collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=labeled_collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=labeled_collate_fn, num_workers=num_workers, pin_memory=True)

    # Model
    model_cfg = cfg.get("model", {})
    num_classes_resolved = get_num_classes(task, target_labels, test_ds)
    model = CuriaClassifier(
        model_repo=model_cfg.get("model_repo", "raidium/curia"),
        local_cache_dir=model_cfg.get("local_cache_dir"),
        token_mode=model_cfg.get("token_mode", "cls"),
        spatial_pool_kernel_size=model_cfg.get("spatial_pool_kernel_size"),
        num_classes=num_classes_resolved,
        num_heads=model_cfg.get("num_heads", 8),
        num_queries=model_cfg.get("num_queries", 1),
        classifier_hidden_dim=model_cfg.get("classifier_hidden_dim", 512),
        classifier_dropout=model_cfg.get("classifier_dropout", 0.5),
        pooling_dropout=model_cfg.get("pooling_dropout", 0.0),
        add_slice_positional_embedding=model_cfg.get("add_slice_positional_embedding", True),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info(f"Parameters: {trainable:,} trainable / {total - trainable:,} frozen / {total:,} total")

    # Optimiser & scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(hp.get("lr", 1e-4)),
        weight_decay=float(hp.get("weight_decay", 0.01)),
    )
    max_epochs = hp.get("max_epochs", 50)
    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(total_steps * hp.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss
    class_weights = (
        compute_class_weights_for_weighted_loss(train_annots, target_labels, task, accelerator.device)
        if hp.get("weighted_loss", False) else None
    )
    criterion = get_loss_criterion(task, class_weights)

    # Output dir
    run_label_tag = build_run_label_tag(task, target_labels, plane)
    output_dir = os.path.join(
        cfg.get("output_dir", "experiments/curia_classifier"),
        f"{dataset_name}_{run_label_tag}_{CURR_TIME}_{JOB_ID}",
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yml"), "w") as f:
            yaml.dump(cfg, f)
        wandb.init(
            project="curia-classifier",
            name=f"{dataset_name}_{run_label_tag}",
            config=cfg,
            dir=output_dir,
        )

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler,
    )

    # Early stopping
    patience = hp.get("patience", 10)
    ckpt_dir = os.path.join(output_dir, "ckpt")
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
    early_stopping = EarlyStopping(
        patience=patience, mode="max",
        ckpt_path=os.path.join(ckpt_dir, "best.pt"),
        accelerator=accelerator,
    ) if patience > 0 else None

    # Training loop
    best_val_auroc = -1.0
    for epoch in range(max_epochs):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, task, criterion, num_classes_resolved, accelerator,
        )
        val_loss, val_metrics, *_ = evaluate(
            model, val_loader, task, criterion, num_classes_resolved, accelerator,
        )
        val_auroc = val_metrics.get("auroc", 0.0)

        if accelerator.is_main_process:
            log_dict = {
                "lr": scheduler.get_last_lr()[0],
                "train/loss": train_loss,
                "val/loss": val_loss,
                "epoch": epoch + 1,
            }
            for k, v in train_metrics.items():
                log_dict[f"train/{k}"] = v
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v
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

    # Validation-derived decision thresholds for threshold-based test metrics.
    val_loss, val_metrics, val_logits, val_labels, _ = evaluate(
        model, val_loader, task, criterion, num_classes_resolved, accelerator,
    )
    val_thresholds = None
    if accelerator.is_main_process:
        if task == "binary":
            val_probs = torch.sigmoid(val_logits.squeeze(-1)).cpu().numpy()
        elif task == "multilabel":
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        else:
            val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy()
        val_thresholds = compute_youden_thresholds(
            y_true=val_labels.cpu().numpy(),
            y_pred_prob=val_probs,
            task=task,
        )

    # Test evaluation
    if accelerator.is_main_process:
        logger.info("Evaluating on test set ...")
    test_loss, test_metrics, test_logits, test_labels, test_ids = evaluate(
        model, test_loader, task, criterion, num_classes_resolved, accelerator,
    )

    if accelerator.is_main_process:
        logger.info(f"Test metrics: {test_metrics}")
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

        # Probabilities
        if task == "binary":
            probs = torch.sigmoid(test_logits.squeeze(-1)).cpu().numpy()
        elif task == "multilabel":
            probs = torch.sigmoid(test_logits).cpu().numpy()
        else:
            probs = torch.softmax(test_logits, dim=-1).cpu().numpy()

        true_np = test_labels.cpu().numpy()

        # Visualise
        viz_labels = class_names if class_names else target_labels
        fig_dir = os.path.join(output_dir, "fig")
        viz_metrics = compute_and_visualize_metrics(
            y_true=true_np, y_pred_prob=probs, task=task,
            class_labels=viz_labels, output_dir=fig_dir, thresholds=val_thresholds,
        )
        logger.info(f"Plots saved to {fig_dir}")

        # Save results
        results = {
            "test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                            for k, v in test_metrics.items()},
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

        wandb.finish()
        logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end Curia classifier (frozen backbone + trainable cross-attention + head)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (see med_slim/configs/curia_classifier.yml).",
    )
    args = parser.parse_args()
    main(args)
