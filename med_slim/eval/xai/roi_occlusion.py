"""
xMIL-style ROI occlusion explanations for the shipped volume classifier.

Explanations are computed on the full ``SingleViewClassifier`` (COBRA + MLP head),
so every score is a real class logit rather than an internal attention weight.
Three experiments run per volume:

A. ROI-group occlusion: drop the annotated lesion window, equally wide lesion-free
   control windows, and everything but the lesion.
B. Leave-one-slice-out: one delta per slice, always measured from the full bag.
C. MoRF insertion and deletion: cumulatively flip slices ranked by the Experiment B
   heatmap, by ABMIL pooling attention, and by a random permutation, then compare
   AUPC to see whether either heatmap is more faithful than chance.

The explained class is always the ground-truth label, so a positive delta means the
perturbed slices supported the correct diagnosis.

Usage:
    python -m med_slim.eval.xai.roi_occlusion \
        --experiment-dir /path/to/experiment_output \
        --fold 3
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from accelerate import Accelerator

from med_slim.data.feat_dataset import FeatClassificationDataset, linear_classifier_collate_fn
from med_slim.eval.load_classifier import load_classifier_from_experiment
from med_slim.eval.xai.drop import abmil_attention, bag_from_batch
from med_slim.eval.xai.faithfulness import MORF_MODES, aupc, mean_curve, morf_curve
from med_slim.eval.xai.occlusion import leave_one_slice_out, roi_group_occlusion
from med_slim.eval.xai.roi_utils import (
    get_roi_slice_range,
    is_positive_pathology_class,
    load_roi_annotations,
    resolve_class_label,
)
from med_slim.utils.label_metadata import (
    build_multiclass_label_names,
    get_annotation_paths_by_split,
    get_dataset_metadata,
)
from med_slim.utils.viz.attention import plot_morf_curves, plot_slice_occlusion_profile
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

RANKINGS = ("occlusion", "abmil", "random")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="xMIL-style ROI occlusion explanations for a COBRA volume classifier"
    )
    parser.add_argument("--experiment-dir", type=str, required=True,
                        help="Linear-probing experiment directory. The MLP head is required, "
                             "so unlike slice attention there is no pretrained-checkpoint mode.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold to load the checkpoint from (e.g. 3 -> fold_3/ckpt/classifier.pt)")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Dataset name (e.g. MRNet, SKM-TEA, kneeMRI)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--feat-dir", type=str, default=None, help="Override the cached feature directory")
    parser.add_argument("--annotations-path", type=str, default=None, help="Override the annotations CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Override the output directory")
    parser.add_argument("--plane", type=str, default=None, help="View plane")
    parser.add_argument("--task", type=str, default=None, choices=["binary", "multiclass"],
                        help="Classification task type. Multilabel is unsupported: a single "
                             "explained class cannot be derived from a multi-hot label.")
    parser.add_argument("--target-labels", type=str, nargs="+", default=None,
                        help="Target label columns overriding `eval_datasets.yaml`")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Must be 1: bags shrink by different amounts per volume.")
    parser.add_argument("--num-samples", type=int, default=None, help="Max volumes to explain (None = all)")
    parser.add_argument("--n-random", type=int, default=5,
                        help="Control windows averaged per volume in Experiment A")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the control windows and the random MoRF ranking")
    parser.add_argument("--morf-points", type=int, default=21,
                        help="Grid size used to average MoRF curves across bag lengths")
    parser.add_argument("--skip-loso", action="store_true",
                        help="Skip Experiment B (also disables the occlusion MoRF ranking)")
    parser.add_argument("--skip-morf", action="store_true", help="Skip Experiment C")
    return parser


def resolve_experiment_inputs(args, cfg: Dict) -> Dict:
    """Resolve feature dir, FM set, annotations, and output dir from the experiment config."""
    feat_cfg = cfg.get("feat_dataset", {})

    plane = args.plane or feat_cfg.get("plane", ["sagittal"])[0]

    if args.feat_dir:
        feat_dir = args.feat_dir
    elif feat_cfg.get("sequences"):
        sequences = feat_cfg["sequences"]
        seq_name = list(sequences.keys())[0]
        feat_dir = sequences[seq_name]
        if len(sequences) > 1:
            logger.warning(
                f"Multiple sequences found ({list(sequences.keys())}); using first: {seq_name}"
            )
    else:
        feat_dir = feat_cfg.get("feat_dir")
    if not feat_dir:
        raise ValueError("Cannot determine the feature directory. Pass --feat-dir explicitly.")

    # The evaluated FM subset is fixed by the checkpoint: the classifier head was
    # trained on exactly these FMs, so K must not be overridden here.
    fm_id_order = cfg.get("fm_id_order") or feat_cfg.get("model_name") or []
    if isinstance(fm_id_order, str):
        fm_id_order = [fm_id_order]
    eval_fm_ids = cfg.get("eval_fm_ids")
    if eval_fm_ids is not None and fm_id_order:
        model_names = [fm_id_order[int(i)] for i in eval_fm_ids]
    else:
        model_names = feat_cfg.get("model_name", ["dinov2"])
        if isinstance(model_names, str):
            model_names = [model_names]

    dataset_name = args.dataset_name or feat_cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("Cannot determine the dataset name. Pass --dataset-name explicitly.")
    ds_meta = get_dataset_metadata(dataset_name)

    task = args.task or cfg.get("task") or ds_meta["task"]
    if task == "multilabel":
        raise ValueError(
            "Occlusion explanations need one explained class per volume, which a "
            "multilabel task does not provide. Explain one label at a time instead."
        )
    target_labels = args.target_labels or ds_meta["target_labels"]

    if args.annotations_path:
        annotations_path = args.annotations_path
    else:
        annotations_dir = cfg.get("annotations_dir")
        if not annotations_dir:
            raise ValueError(
                "Cannot determine annotations_dir from the experiment config. "
                "Pass --annotations-path explicitly."
            )
        annot_files = get_annotation_paths_by_split(annotations_dir, task, [args.split])
        annotations_path = str(annot_files[args.split])

    if args.output_dir:
        output_dir = args.output_dir
    elif args.fold is not None:
        output_dir = os.path.join(args.experiment_dir, f"fold_{args.fold}", "roi_occlusion")
    else:
        output_dir = os.path.join(args.experiment_dir, "roi_occlusion")

    multiclass_names = None
    if task == "multiclass":
        multiclass_names = build_multiclass_label_names(
            target_labels[0], ds_meta.get("multiclass_label_maps")
        )

    return {
        "plane": plane,
        "feat_dir": feat_dir,
        "model_names": model_names,
        "dataset_name": dataset_name,
        "task": task,
        "target_labels": target_labels,
        "annotations_path": annotations_path,
        "output_dir": output_dir,
        "display_map": ds_meta.get("label_display_names"),
        "multiclass_names": multiclass_names,
    }


def log_group_summary(rows: List[Dict], title: str) -> None:
    """Log mean Experiment A deltas for a subset of volumes."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    logger.info(f"--- {title} ({len(df)} volumes) ---")
    logger.info(
        f"  delta_logit  ROI: {df['delta_logit_roi'].mean():+.4f}  "
        f"random: {df['delta_logit_random'].mean():+.4f}  "
        f"complement: {df['delta_logit_complement'].mean():+.4f}"
    )
    logger.info(
        f"  delta_prob   ROI: {df['delta_prob_roi'].mean():+.4f}  "
        f"random: {df['delta_prob_random'].mean():+.4f}  "
        f"complement: {df['delta_prob_complement'].mean():+.4f}"
    )
    wins = float((df["delta_logit_roi"] > df["delta_logit_random"]).mean())
    logger.info(
        f"  ROI beats random on {wins:.1%} of volumes "
        f"(mean gain {df['roi_gain_over_random'].mean():+.4f})"
    )


def log_morf_summary(rows: List[Dict], title: str) -> None:
    """Log mean AUPC per ranking and mode for a subset of volumes."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    logger.info(f"--- {title} ({df['sample_id'].nunique()} volumes) ---")
    for mode in MORF_MODES:
        direction = "lower is better" if mode == "drop" else "higher is better"
        parts = []
        for ranking in RANKINGS:
            subset = df[(df["mode"] == mode) & (df["ranking"] == ranking)]
            if not subset.empty:
                parts.append(f"{ranking}: {subset['aupc'].mean():.4f}")
        logger.info(f"  AUPC {mode:<5} ({direction})  " + "  ".join(parts))


def main():
    args = build_parser().parse_args()
    if args.batch_size != 1:
        build_parser().error(
            "--batch-size must be 1: slice drops shorten each bag by a different amount."
        )

    accelerator = Accelerator()
    model, cfg = load_classifier_from_experiment(args.experiment_dir, accelerator, fold=args.fold)
    model = model.to(accelerator.device)
    model.eval()

    inputs = resolve_experiment_inputs(args, cfg)
    os.makedirs(inputs["output_dir"], exist_ok=True)
    profile_dir = os.path.join(inputs["output_dir"], "occlusion_profiles")

    roi_df = load_roi_annotations(inputs["annotations_path"])
    if roi_df is None:
        raise ValueError(
            f"{inputs['annotations_path']} has no roiZ / roiDepth columns, so there is "
            "no ROI to occlude."
        )

    logger.info("=" * 60)
    logger.info("ROI Occlusion Explanations (xMIL-style slice drop)")
    logger.info("=" * 60)
    logger.info(f"Experiment dir: {args.experiment_dir}  fold={args.fold}")
    logger.info(f"Feature dir: {inputs['feat_dir']}")
    logger.info(f"Eval FMs (K={len(inputs['model_names'])}): {inputs['model_names']}")
    logger.info(f"Plane: {inputs['plane']}, Split: {args.split}")
    logger.info(f"Task: {inputs['task']}, Target labels: {inputs['target_labels']}")
    logger.info("=" * 60)

    dataset = FeatClassificationDataset(
        feat_dir=inputs["feat_dir"],
        slice_encoder_models=inputs["model_names"],
        view_plane=inputs["plane"],
        split=args.split,
        annotations_path=inputs["annotations_path"],
        task=inputs["task"],
        target_columns=inputs["target_labels"],
    )
    logger.info(f"Loaded {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=linear_classifier_collate_fn,
        num_workers=4,
    )

    rng = np.random.default_rng(args.seed)
    group_rows: List[Dict] = []
    slice_rows: List[Dict] = []
    morf_rows: List[Dict] = []
    # (is_tear, curve) per volume, so the tear-only mean curve needs no re-pairing.
    morf_curves: Dict[str, Dict[str, List[Tuple[bool, Dict]]]] = {
        mode: {ranking: [] for ranking in RANKINGS} for mode in MORF_MODES
    }
    n_profiles = 0
    n_no_roi = 0

    for i, batch in enumerate(dataloader):
        if args.num_samples is not None and i >= args.num_samples:
            break

        sample_id = str(batch["sample_ids"][0])
        label = batch["labels"][0]
        class_label = resolve_class_label(
            inputs["task"], label, inputs["target_labels"],
            inputs["display_map"], inputs["multiclass_names"],
        )
        is_tear = is_positive_pathology_class(class_label)
        class_index = int(label.item())

        bag = bag_from_batch(batch, accelerator.device)
        seq_len = bag.seq_len
        roi_range = get_roi_slice_range(roi_df, sample_id, seq_len)
        gt_span = (
            (roi_range["roi_start_slice"], roi_range["roi_end_slice"])
            if roi_range is not None else None
        )

        # Experiment A: ROI vs control windows vs ROI-only
        full_scores: Optional[Dict[str, float]] = None
        if roi_range is not None:
            row = roi_group_occlusion(
                model, bag, roi_range, class_index, rng, n_random=args.n_random
            )
            row.update({
                "sample_id": sample_id,
                "class_label": class_label,
                "class_index": class_index,
                "is_tear": int(is_tear),
            })
            group_rows.append(row)
            full_scores = {"logit": row["full_logit"], "prob": row["full_prob"]}
        else:
            n_no_roi += 1

        # Experiment B: leave-one-slice-out heatmap
        loso = None
        if not args.skip_loso:
            loso = leave_one_slice_out(model, bag, class_index, full_scores=full_scores)
            full_scores = loso["full"]

        attention = abmil_attention(model, bag)

        if loso is not None:
            for t in range(seq_len):
                slice_rows.append({
                    "sample_id": sample_id,
                    "class_label": class_label,
                    "slice_index": t,
                    "seq_len": seq_len,
                    "in_roi": int(gt_span is not None and gt_span[0] <= t <= gt_span[1]),
                    "delta_logit": float(loso["delta_logit"][t]),
                    "delta_prob": float(loso["delta_prob"][t]),
                    "abmil_attention": float(attention[t]),
                })
            if is_tear:
                plot_slice_occlusion_profile(
                    delta_logit=loso["delta_logit"],
                    sample_id=sample_id,
                    view_plane=inputs["plane"],
                    output_dir=profile_dir,
                    ground_truth_range=gt_span,
                    class_label=class_label,
                )
                n_profiles += 1

        # Experiment C: MoRF deletion / insertion for each ranking
        if not args.skip_morf:
            ranking_scores = {
                "abmil": attention,
                "random": rng.permutation(seq_len).astype(np.float64),
            }
            if loso is not None:
                ranking_scores["occlusion"] = loso["delta_logit"]
            for mode in MORF_MODES:
                for ranking, scores in ranking_scores.items():
                    curve = morf_curve(model, bag, scores, class_index, mode=mode)
                    morf_curves[mode][ranking].append((is_tear, curve))
                    morf_rows.append({
                        "sample_id": sample_id,
                        "class_label": class_label,
                        "is_tear": int(is_tear),
                        "seq_len": seq_len,
                        "mode": mode,
                        "ranking": ranking,
                        "aupc": aupc(curve["fractions"], curve["probs"]),
                        "probs": " ".join(f"{p:.6f}" for p in curve["probs"]),
                        "order": " ".join(str(int(t)) for t in curve["order"]),
                    })

        if (i + 1) % 10 == 0:
            logger.info(f"Explained {i + 1} volumes")

    logger.info("=" * 60)
    logger.info("Occlusion Summary")
    logger.info("=" * 60)
    if n_no_roi:
        logger.info(f"{n_no_roi} volumes had no usable roiZ / roiDepth and skipped Experiment A")

    if group_rows:
        group_df = pd.DataFrame(group_rows)
        group_csv = os.path.join(inputs["output_dir"], "roi_group_occlusion.csv")
        group_df.to_csv(group_csv, index=False)
        log_group_summary([r for r in group_rows if r["is_tear"]], "Experiment A (tear only)")
        log_group_summary(group_rows, "Experiment A (all classes)")
        logger.info(f"Saved ROI-group occlusion to {group_csv}")

    if slice_rows:
        slice_df = pd.DataFrame(slice_rows)
        slice_csv = os.path.join(inputs["output_dir"], "slice_scores_occlusion.csv")
        slice_df.to_csv(slice_csv, index=False)
        logger.info(f"Saved {n_profiles} leave-one-slice-out profiles for tear cases")
        logger.info(f"Saved per-slice occlusion scores to {slice_csv}")

    if morf_rows:
        morf_df = pd.DataFrame(morf_rows)
        morf_csv = os.path.join(inputs["output_dir"], "morf_faithfulness.csv")
        morf_df.to_csv(morf_csv, index=False)
        log_morf_summary([r for r in morf_rows if r["is_tear"]], "Experiment C (tear only)")
        log_morf_summary(morf_rows, "Experiment C (all classes)")
        logger.info(f"Saved MoRF faithfulness to {morf_csv}")

        n_tear = len({r["sample_id"] for r in morf_rows if r["is_tear"]})
        for mode in MORF_MODES:
            curves = {}
            for ranking in RANKINGS:
                tear_curves = [c for is_tear, c in morf_curves[mode][ranking] if is_tear]
                if tear_curves:
                    curves[ranking] = mean_curve(tear_curves, num_points=args.morf_points)
            if curves:
                plot_morf_curves(
                    curves,
                    output_dir=inputs["output_dir"],
                    mode=mode,
                    title=(
                        f'MoRF {"deletion" if mode == "drop" else "insertion"} '
                        f'({n_tear} tear volumes)'
                    ),
                )

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
