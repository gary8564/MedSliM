#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=1:00:00                 
#SBATCH --job-name=kneeMRI_linear_probing_%j
#SBATCH --output=stdout_lp_kneeMRI_%j.txt    
#SBATCH --account=p0021834    

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CONFIG_PATH="./med_slim/configs/linear_classifier.yml"
CHECKPOINT_PATH="/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-03-15-04:23/medslim-epoch3000.pth.tar" #"/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet/2026-02-07-14:10/medslim-epoch2000.pth.tar"  # Leave empty to use config file, or set path to override

# Feature cache must match pretrained model checkpoint:
#   global-only pretrain  -> .../slices_raw/crop
#   tiled (regional_tokens=4) -> .../slices_raw/crop_tiled_2x2
FEAT_DIR="/hpcwork/rwth1833/feat_caches/kneeMRI/slices_raw/crop"

FINE_TUNE=false  # whether to fine-tune COBRA backbone
FM_POOLING="avg_pool"  # Options: "avg_pool", "attention" (attention requires fine-tuning COBRA)
SEQUENCE_ENCODER="mamba2"
FM_MODEL_NAMES="mri-core medimageinsight curia"
# Global-only ABMIL: raw / post_embed / post_encoder. Tiled ABMIL: post_embed / post_encoder.
# Leave empty to auto-resolve from checkpoint (abmil + global -> raw, else -> post_embed).
POOLING_TARGET="raw"
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

if [[ -n "${POOLING_TARGET}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --pooling-target ${POOLING_TARGET}"
fi

### Run script
echo "Starting linear classifier evaluation..."

python ./med_slim/eval/linear_classifier.py \
  --linear-classifier-config "${CONFIG_PATH}" \
  --feat-dir "${FEAT_DIR}" \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --weighted-loss \
  --fm-pooling "${FM_POOLING}" \
  --sequence-encoder "${SEQUENCE_ENCODER}" \
  --n-folds ${N_FOLDS} \
  --train-fraction "${TRAIN_FRACTION}" \
  --num-repeats "${NUM_REPEATS}" \
  ${EXTRA_ARGS}

echo "Linear probing evaluation complete!"