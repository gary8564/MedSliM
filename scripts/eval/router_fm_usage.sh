#!/usr/bin/bash

### Job Parameters
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=c23g
#SBATCH --time=01:00:00
#SBATCH --job-name=router_fm_usage_%j
#SBATCH --output=logs/eval/stdout_router_fm_usage_%j.txt
#SBATCH --account=p0021834

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet/2026-06-16-17:51/medslim-epoch2000.pth.tar}"
# CONFIG=""                 # Optional: override pretrain config (default: beside checkpoint or pretrain.yml)
ENCODER="momentum"          # momentum (LP/COBRA default) | base
OUTPUT="reports/fm_usage/router_usage_2026-06-16-17:51.png"
NUM_BATCHES=100
BATCH_SIZE=""               # Leave empty for script default (min(64, train.batch_size))
NUM_WORKERS=16
# FM_NAMES=""               # Optional: space-separated subset; default = all pretrain FMs

### Build extra args
EXTRA_ARGS=""
if [[ -n "${CONFIG}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --config ${CONFIG}"
fi
if [[ -n "${ENCODER}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --encoder ${ENCODER}"
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --batch-size ${BATCH_SIZE}"
fi
if [[ -n "${NUM_WORKERS}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --num-workers ${NUM_WORKERS}"
fi
if [[ -n "${FM_NAMES}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --fm-names ${FM_NAMES}"
fi
if [[ -n "${OUTPUT}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --output ${OUTPUT}"
fi

### Run
set -euo pipefail
mkdir -p logs/eval reports/fm_usage
echo "Starting router FM usage analysis ..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Encoder: ${ENCODER}"

python -m med_slim.eval.router_fm_usage \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num-batches ${NUM_BATCHES} \
  ${EXTRA_ARGS}

echo "Router FM usage analysis complete!"
