#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=2:00:00                 
#SBATCH --job-name=kneemri_linear_probing_%j
#SBATCH --output=stdout_kneemri_linear_probing_%j.txt    
#SBATCH --account=p0021834    

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CONFIG_PATH="./med_slim/configs/linear_classifier.yml"
CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-06-23-00:12/medslim-epoch2000.pth.tar "  # Leave empty to use config file, or set path to override
FINE_TUNE=true  # whether to fine-tune COBRA backbone
FM_POOLING="router"  # Options: "avg_pool", "router"; empty = infer from checkpoint
SEQUENCE_ENCODER="mamba2"  # mamba2 or transformer; empty = infer from checkpoint
FM_MODEL_NAMES="mri-core medimageinsight curia"
POOLING_TARGET="post_encoder" #"raw" "post_encoder"
N_FOLDS=3
SLICE_POOLING=""  # leave empty to auto-detect from checkpoint
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

### Run script
echo "Starting linear classifier evaluation..."

python ./med_slim/eval/linear_classifier.py \
  --linear-classifier-config "${CONFIG_PATH}" \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --weighted-loss \
  --pooling-target "${POOLING_TARGET}" \
  --n-folds ${N_FOLDS} \
  ${EXTRA_ARGS}

echo "Linear probing evaluation complete!"

