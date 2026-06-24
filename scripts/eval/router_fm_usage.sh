#!/usr/bin/bash

### Job Parameters
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=c23g
#SBATCH --time=02:00:00
#SBATCH --job-name=router_fm_usage_%j
#SBATCH --output=logs/eval/stdout_router_fm_usage_%j.txt
#SBATCH --account=p0021834

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-06-21-18:27/medslim-epoch2000.pth.tar"
# CONFIG=""                 # Optional: override pretrain config (default: beside checkpoint or pretrain.yml)
OUTPUT="reports/fm_usage/router_T_0.5_confidence_0.001.png"
NUM_BATCHES=200
BATCH_SIZE=""               # Leave empty to use train.batch_size from config
NUM_WORKERS=16              # More workers speed up on-demand safetensors reads
CACHE_IN_MEMORY=false       # true: preload all features into RAM; useful for many batches

### Build extra args
EXTRA_ARGS=""
if [[ -n "${CONFIG}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --config ${CONFIG}"
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --batch-size ${BATCH_SIZE}"
fi
if [[ -n "${NUM_WORKERS}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --num-workers ${NUM_WORKERS}"
fi
if [[ "${CACHE_IN_MEMORY}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --cache-in-memory"
fi
if [[ -n "${OUTPUT}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --output ${OUTPUT}"
fi

### Run
mkdir -p logs/eval
echo "Starting router FM usage analysis ..."
echo "Checkpoint: ${CHECKPOINT_PATH}"

python -m med_slim.eval.router_fm_usage \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num-batches ${NUM_BATCHES} \
  ${EXTRA_ARGS}

echo "Router FM usage analysis complete!"
