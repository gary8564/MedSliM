#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=1:00:00                 
#SBATCH --job-name=lp_binary_classification_%j
#SBATCH --output=stdout_lp_binary_classification_%j.txt    
#SBATCH --account=p0021834    

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CONFIG_PATH="./med_slim/configs/linear_classifier.yml"
CHECKPOINT_PATH="/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/test-run-MRNet/2026-01-11-15:34/medslim-epoch2000.pth.tar"  # Leave empty to use config file, or set path to override
FINE_TUNE=false  # whether to fine-tune COBRA backbone
FM_POOLING="avg_pool"  # Options: "avg_pool", "attention" (attention requires fine-tuning COBRA)
SEQUENCE_ENCODER="mamba2"
SLICE_POOLING="cls" # Only used when sequence encoder is transformer
FM_MODEL_NAMES="dinov2 dinov3 rad-dino medsiglip biomedclip ark" 
POOLING_TARGET="post_embed"

EXTRA_ARGS=""
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --checkpoint-path ${CHECKPOINT_PATH}"
fi

if [[ "${FINE_TUNE}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --fine-tune"
fi

# Only pass --slice-pooling for transformer
if [[ "${SEQUENCE_ENCODER}" == "transformer" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --slice-pooling ${SLICE_POOLING}"
fi

### Run script
echo "Starting linear classifier evaluation..."

python ./med_slim/eval/linear_classifier.py \
  --linear-classifier-config "${CONFIG_PATH}" \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --weighted-loss \
  --fm-pooling "${FM_POOLING}" \
  --sequence-encoder "${SEQUENCE_ENCODER}" \
  --pooling-target "${POOLING_TARGET}" \
  ${EXTRA_ARGS}

echo "Linear probing evaluation complete!"

