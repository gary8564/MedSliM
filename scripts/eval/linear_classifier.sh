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
#SBATCH --output=stdout_SKM-TEA_linear_probing_%j.txt
#SBATCH --partition=c23g
#SBATCH --account=p0021834
# Depends on still-running precompute jobs only, e.g.:
#   sbatch --dependency=afterok:1731051:1731053:1731055:1731057 scripts/eval/linear_classifier.sh

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CONFIG_PATH="./med_slim/configs/linear_classifier.yml"
CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-03-15-04:23/medslim-epoch3000.pth.tar" #"/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet/2026-02-07-14:10/medslim-epoch2000.pth.tar"  # Leave empty to use config file, or set path to override
# CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet/2026-06-16-17:51/medslim-epoch2000.pth.tar"
# CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-06-21-03:09/medslim-epoch2000.pth.tar"
# CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K-OAI/2026-07-19-12:30:24_2062384/medslim-epoch2000.pth.tar"

# Feature cache must match pretrained model checkpoint:
#   global-only pretrain  -> .../slices_raw/crop
#   tiled (regional_tokens=4) -> .../slices_raw/crop_tiled_2x2
FEAT_DIR="/hpcwork/rwth1833/feat_caches/SKM-TEA/DESS_E1/slices_raw/adaptive"

FINE_TUNE=false  # whether to fine-tune COBRA backbone
FM_POOLING="avg_pool"  # Options: "avg_pool", "router" (router requires a router-pretrained checkpoint)
SEQUENCE_ENCODER="mamba2"
FM_MODEL_NAMES="mri-core"
# ABMIL: raw / post_embed / post_encoder.
# Global-only ABMIL: raw / post_embed / post_encoder. Tiled ABMIL: post_embed / post_encoder.
POOLING_TARGET="raw"
# Required for raw pooling when FM_MODEL_NAMES contains multiple FMs. Must specify one of the FMs in FM_MODEL_NAMES.
# Leave empty only for single-FM evaluation or when POOLING_TARGET is not raw.
RAW_AGGREGATION_FM="mri-core"
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

python ./med_slim/eval/linear_classifier.py \
  --linear-classifier-config "${CONFIG_PATH}" \
  --feat-dir "${FEAT_DIR}" \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --weighted-loss \
  --n-folds ${N_FOLDS} \
  --train-fraction "${TRAIN_FRACTION}" \
  --num-repeats "${NUM_REPEATS}" \
  ${EXTRA_ARGS}

echo "Linear probing evaluation complete!"