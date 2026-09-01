#!/usr/bin/bash

### Job Parameters
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=medslim_baseline_anymc3d_classifier_%j
#SBATCH --output=logs/eval/stdout_anymc3d_classifier_%j.txt

### Setup
set -e
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
# Frozen 2D FM cache + AnyMC3D task-query pooling. See baselines/anymc3d/README.md.
# Same 3-fold-on-train / official-test protocol as scripts/eval/linear_classifier.sh.
CONFIG_PATH="${CONFIG_PATH:-./baselines/anymc3d/configs/kneeMRI.yml}"
FEAT_DIR="${FEAT_DIR:-/hpcwork/rwth1833/feat_caches/kneeMRI/slices_raw/crop}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-/hpcwork/rwth1833/datasets/preprocessed/kneeMRI}"
OUTPUT_DIR="${OUTPUT_DIR:-/hpcwork/rwth1833/experiments/anymc3d-classifier}"
MODEL_NAME="${MODEL_NAME:-mri-core}"
PLANE="${PLANE:-sagittal}"
WEIGHTED_LOSS="${WEIGHTED_LOSS:-true}"
N_FOLDS="${N_FOLDS:-3}"
SEED="${SEED:-42}"

EXTRA_ARGS=""
if [[ "${WEIGHTED_LOSS}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --weighted-loss"
fi

### Run
echo "Starting AnyMC3D baseline classifier (cached 2D FM features) ..."
echo "  config:      ${CONFIG_PATH}"
echo "  feat_dir:    ${FEAT_DIR}"
echo "  annots:      ${ANNOTATIONS_DIR}"
echo "  output_dir:  ${OUTPUT_DIR}"
echo "  model:       ${MODEL_NAME}"
echo "  plane:       ${PLANE}"
echo "  weighted:    ${WEIGHTED_LOSS}"
echo "  n_folds:     ${N_FOLDS}"
echo "  seed:        ${SEED}"

python -m baselines.anymc3d.classifier \
  --config "${CONFIG_PATH}" \
  --feat-dir "${FEAT_DIR}" \
  --annotations-dir "${ANNOTATIONS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --plane "${PLANE}" \
  --n-folds "${N_FOLDS}" \
  --seed "${SEED}" \
  ${EXTRA_ARGS}

echo "AnyMC3D baseline evaluation complete!"
