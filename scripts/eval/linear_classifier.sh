#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=01:00:00                 
#SBATCH --job-name=SKM-TEA_linear_probing_%j
#SBATCH --output=logs/eval/stdout_SKM-TEA_linear_probing_%j.txt
#SBATCH --partition=c23g
#SBATCH --account=p0021834
# Depends on still-running precompute jobs only, e.g.:
#   sbatch --dependency=afterok:1731051:1731053:1731055:1731057 scripts/eval/linear_classifier.sh

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CONFIG_PATH="./med_slim/configs/linear_classifier.yml"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-03-15-04:23/medslim-epoch3000.pth.tar}"
# CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet/2026-06-16-17:51/medslim-epoch2000.pth.tar"
# CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-06-21-03:09/medslim-epoch2000.pth.tar"
# CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K-OAI/2026-07-19-12:30:24_2062384/medslim-epoch2000.pth.tar"

# Feature cache must match pretrained model checkpoint:
#   global-only pretrain  -> .../slices_raw/crop
#   tiled (regional_tokens=4) -> .../slices_raw/crop_tiled_2x2
DATASET_NAME="${DATASET_NAME:-SKM-TEA}"
FEAT_DIR="${FEAT_DIR:-/hpcwork/rwth1833/feat_caches/SKM-TEA/DESS_E1/slices_raw/adaptive}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-/hpcwork/rwth1833/datasets/preprocessed/SKM-TEA}"
OUTPUT_DIR="${OUTPUT_DIR:-/hpcwork/qj474765/experiments/MedSliM-linear-probing/SKM-TEA}"
PLANES="${PLANES:-sagittal}"
# RSNA-Knee override example:
#   DATASET_NAME=RSNA-Knee \
#   FEAT_DIR=/hpcwork/rwth1833/feat_caches/RSNA-Knee/slices_raw/adaptive \
#   ANNOTATIONS_DIR=/hpcwork/rwth1833/datasets/preprocessed/RSNA-Knee \
#   OUTPUT_DIR=/hpcwork/rwth1833/experiments/MedSliM-linear-probing/RSNA-Knee \
#   PLANES="sagittal coronal axial" \
#   bash scripts/eval/linear_classifier.sh

FINE_TUNE="${FINE_TUNE:-false}"
# COBRA modules to unfreeze. Choices: attn embed seq_enc fm_router proj norm all
# TRAINABLE_LAYERS set → unfreeze those modules (FINE_TUNE is optional).
# FINE_TUNE=true with TRAINABLE_LAYERS empty → unfreeze all.
# Neither → linear probing.
# Examples: "attn"  |  "attn fm_router"  |  "all"
TRAINABLE_LAYERS="${TRAINABLE_LAYERS:-}"
FM_POOLING="${FM_POOLING:-avg_pool}"  # Options: "avg_pool", "router" (router requires a router-pretrained checkpoint)
SEQUENCE_ENCODER="mamba2"
FM_MODEL_NAMES="${FM_MODEL_NAMES:-mri-core}"
# ABMIL: raw / post_embed / post_encoder.
# Global-only ABMIL: raw / post_embed / post_encoder. Tiled ABMIL: post_embed / post_encoder.
POOLING_TARGET="raw"
# Required for raw pooling when FM_MODEL_NAMES contains multiple FMs. Must specify one of the FMs in FM_MODEL_NAMES.
# Leave empty only for single-FM evaluation or when POOLING_TARGET is not raw.
RAW_AGGREGATION_FM="${RAW_AGGREGATION_FM:-mri-core}"
N_FOLDS=3
SLICE_POOLING=""  # leave empty to auto-detect from checkpoint

# Few-shot / repeated-evaluation overrides (env-overridable for sweeps).
# TRAIN_FRACTION: label budget in (0, 1]; 1.0 = full training set.
# NUM_REPEATS: repeated random runs for mean +/- std uncertainty.
TRAIN_FRACTION=${TRAIN_FRACTION:-1.0}
NUM_REPEATS=${NUM_REPEATS:-1}

EXTRA_ARGS=""
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --checkpoint-path ${CHECKPOINT_PATH}"
fi

if [[ "${FINE_TUNE}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --fine-tune"
fi

if [[ -n "${TRAINABLE_LAYERS}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --trainable-layers ${TRAINABLE_LAYERS}"
fi

if [[ -n "${SLICE_POOLING}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --slice-pooling ${SLICE_POOLING}"
fi

if [[ -n "${FM_POOLING}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --fm-pooling ${FM_POOLING}"
fi

if [[ -n "${SEQUENCE_ENCODER}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --sequence-encoder ${SEQUENCE_ENCODER}"
fi

if [[ -n "${POOLING_TARGET}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --pooling-target ${POOLING_TARGET}"
fi

if [[ -n "${RAW_AGGREGATION_FM}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --raw-aggregation-fm ${RAW_AGGREGATION_FM}"
fi

### Run script
echo "Starting linear classifier evaluation..."
echo "  dataset:    ${DATASET_NAME}"
echo "  checkpoint: ${CHECKPOINT_PATH:-<from config>}"
echo "  feat_dir:   ${FEAT_DIR}"
echo "  annots:     ${ANNOTATIONS_DIR}"
echo "  planes:     ${PLANES}"
echo "  fm_pooling: ${FM_POOLING}  pooling_target: ${POOLING_TARGET}"
echo "  fm_models:  ${FM_MODEL_NAMES}"
echo "  raw_agg_fm: ${RAW_AGGREGATION_FM}"
if [[ -n "${TRAINABLE_LAYERS}" ]]; then
  EFFECTIVE_LAYERS="${TRAINABLE_LAYERS}"
elif [[ "${FINE_TUNE}" == "true" ]]; then
  EFFECTIVE_LAYERS="all"
else
  EFFECTIVE_LAYERS=""
fi
if [[ -n "${EFFECTIVE_LAYERS}" ]]; then
  echo "  mode:       fine-tune  trainable_layers: ${EFFECTIVE_LAYERS}"
else
  echo "  mode:       linear probing (frozen COBRA)"
fi

python ./med_slim/eval/linear_classifier.py \
  --linear-classifier-config "${CONFIG_PATH}" \
  --dataset-name "${DATASET_NAME}" \
  --feat-dir "${FEAT_DIR}" \
  --annotations-dir "${ANNOTATIONS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --planes ${PLANES} \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --weighted-loss \
  --n-folds ${N_FOLDS} \
  --train-fraction "${TRAIN_FRACTION}" \
  --num-repeats "${NUM_REPEATS}" \
  ${EXTRA_ARGS}

echo "Linear probing evaluation complete!"
