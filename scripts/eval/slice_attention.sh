#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=1:00:00                 
#SBATCH --job-name=slice_attention_%j
#SBATCH --output=stdout_slice_attention_%j.txt    
#SBATCH --account=p0021834    

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
# kneeMRI only: ROI-based slice attention metrics require roiZ/roiDepth annotations.
# CHECKPOINT_PATH="/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet-KMAR50K/2026-02-15-22:14/medslim-epoch2000.pth.tar"
# FEAT_DIR="/hpcwork/rwth1833/feat_caches/kneeMRI/slices_raw/crop"
# ANNOTATIONS_PATH="/hpcwork/rwth1833/datasets/preprocessed/kneeMRI/test_multiclass.csv"
# OUTPUT_DIR="/hpcwork/rwth1833/experiments/MedSliM-linear-probing/slice_attention_kneeMRI"
# DATASET_NAME="kneeMRI"
# PLANE="sagittal"
# FM_MODEL_NAMES="mri-core"

# Recommended future mode once experiment configs include cobra_config:
EXPERIMENT_DIR="/hpcwork/rwth1833/experiments/MedSliM-linear-probing/acl_sagittal_2026-03-09-01:15"

# Visualization settings
SPLIT="test"
BATCH_SIZE=32
NUM_SAMPLES=""          # Leave empty for all samples, or set number (e.g., "50")
PER_HEAD=true          # Set to true for per-head attention profiles

### Build extra args
EXTRA_ARGS=""

if [[ -n "${NUM_SAMPLES}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --num-samples ${NUM_SAMPLES}"
fi

if [[ "${PER_HEAD}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --per-head"
fi

### Run script
echo "Starting COBRA slice attention visualization ..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Dataset: ${DATASET_NAME}"

# python -m med_slim.eval.slice_attention \
#   --checkpoint-path "${CHECKPOINT_PATH}" \
#   --feat-dir "${FEAT_DIR}" \
#   --annotations-path "${ANNOTATIONS_PATH}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --dataset-name "${DATASET_NAME}" \
#   --plane "${PLANE}" \
#   --fm-model-names "${FM_MODEL_NAMES}" \
#   --split "${SPLIT}" \
#   --batch-size ${BATCH_SIZE} \
#   ${EXTRA_ARGS}

# Future experiment-dir mode:
python -m med_slim.eval.slice_attention \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --batch-size ${BATCH_SIZE} \
  ${EXTRA_ARGS}

echo "Slice attention visualization complete!"
echo "Output saved under: ${OUTPUT_DIR}"
