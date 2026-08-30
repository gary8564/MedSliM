#!/usr/bin/bash

### Job Parameters
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=36:00:00
#SBATCH --job-name=curia_classifier_%j
#SBATCH --output=logs/eval/stdout_curia_classifier_%j.txt
#SBATCH --account=p0021834

### Setup
set -e
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Curia weights are gated (RAIL-M). Accept the license at
# https://huggingface.co/raidium/curia and export HF_TOKEN before submitting.

### Configuration
# Official Curia baseline. See baselines/curia/README.md.
# First run builds the token cache; later runs reuse it.
# Same 3-fold-on-train / official-test protocol as scripts/eval/linear_classifier.sh.
CONFIG_PATH="${CONFIG_PATH:-./baselines/curia/configs/kneeMRI.yml}"
DATA_DIR="${DATA_DIR:-/hpcwork/rwth1833/datasets/preprocessed/kneeMRI}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-/hpcwork/rwth1833/datasets/preprocessed/kneeMRI}"
OUTPUT_DIR="${OUTPUT_DIR:-/hpcwork/rwth1833/experiments/curia-classifier}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-/hpcwork/rwth1833/feat_caches/curia_official}"
WEIGHTED_LOSS="${WEIGHTED_LOSS:-true}"
N_FOLDS="${N_FOLDS:-3}"
SEED="${SEED:-42}"

EXTRA_ARGS=""
if [[ "${WEIGHTED_LOSS}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --weighted-loss"
fi

### Run
echo "Starting Curia baseline classifier (cached tokens) ..."
echo "  config:      ${CONFIG_PATH}"
echo "  data_dir:    ${DATA_DIR}"
echo "  annots:      ${ANNOTATIONS_DIR}"
echo "  output_dir:  ${OUTPUT_DIR}"
echo "  token cache: ${FEATURE_CACHE_DIR}"
echo "  weighted:    ${WEIGHTED_LOSS}"
echo "  n_folds:     ${N_FOLDS}"
echo "  seed:        ${SEED}"

python -m baselines.curia.classifier \
  --config "${CONFIG_PATH}" \
  --data-dir "${DATA_DIR}" \
  --annotations-dir "${ANNOTATIONS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --feature-cache-dir "${FEATURE_CACHE_DIR}" \
  --n-folds "${N_FOLDS}" \
  --seed "${SEED}" \
  ${EXTRA_ARGS}

echo "Curia baseline evaluation complete!"