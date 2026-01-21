#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=4:00:00                 
#SBATCH --job-name=lp_binary_classification_meniscus_%j
#SBATCH --output=stdout_lp_binary_classification_meniscus_%j.txt    
#SBATCH --account=rwth1833    

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CONFIG_PATH="./med_slim/configs/linear_classifier.yml"
CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/test-run-MRNet/2026-01-11-14:58/medslim-epoch2000.pth.tar"  # Leave empty to use config file, or set path to override
FM_POOLING="mean"  # Options: "mean", "concat" (attention requires fine-tuning COBRA)
SEQUENCE_ENCODER="transformer"
SLICE_POOLING="cls"
FM_MODEL_NAMES="dinov2 medsiglip ark"

EXTRA_ARGS=""
if [ -n "${CHECKPOINT_PATH}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --checkpoint-path ${CHECKPOINT_PATH}"
fi

### Run script
echo "Starting linear classifier evaluation..."

python ./med_slim/eval/linear_classifier.py \
  --linear-classifier-config "${CONFIG_PATH}" \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --weighted-loss \
  --fm-pooling "${FM_POOLING}" \
  --sequence-encoder "${SEQUENCE_ENCODER}" \
  --slice-pooling "${SLICE_POOLING}" \
  ${EXTRA_ARGS}

echo "Linear probing evaluation complete!"

