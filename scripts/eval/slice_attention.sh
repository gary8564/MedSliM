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
#SBATCH --output=logs/eval/stdout_slice_attention_%j.txt    
#SBATCH --account=p0021834    

### Setup
set -e
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration — choose ONE mode below
# MODE 1: experiment-dir (fine-tuned / LP COBRA from linear_classifier.py).
# Loads fold_*/ckpt/classifier.pt (includes FT ABMIL). Empty EXPERIMENT_DIR → MODE 2.
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/hpcwork/rwth1833/experiments/MedSliM-linear-probing/meniscal_tear_ligament_tear_cartilage_lesion_effusion_DESS_E2/sagittal_2026-03-17-01:46_65920427}"
FOLD=3                  # Which fold's classifier.pt to load (1, 2, or 3)
DATASET_NAME="${DATASET_NAME:-SKM-TEA}"

# MODE 2: explicit checkpoint (pretrained COBRA, no fine-tuning)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-02-15-11:55/medslim-epoch2000.pth.tar}"
FEAT_DIR="${FEAT_DIR:-/hpcwork/rwth1833/feat_caches/MRNet/slices_raw/crop}"
ANNOTATIONS_PATH="${ANNOTATIONS_PATH:-/hpcwork/rwth1833/datasets/preprocessed/MRNet/test.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/hpcwork/rwth1833/experiments/MedSliM-linear-probing/slice_attention_MRNet}"
PLANE="${PLANE:-sagittal}"
FM_MODEL_NAMES="${FM_MODEL_NAMES:-mri-core}"

# Visualization settings
SPLIT="test"
BATCH_SIZE=8
NUM_SAMPLES=""          # Leave empty for all samples, or set number (e.g., "50")
PER_HEAD=true           # Set to true for per-head attention profiles

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

if [[ -n "${EXPERIMENT_DIR}" ]]; then
  # MODE 1: fine-tuned / LP model from experiment directory
  echo "Experiment dir: ${EXPERIMENT_DIR}"
  echo "Fold: ${FOLD}"
  echo "Dataset: ${DATASET_NAME}"

  python -m med_slim.eval.xai.slice_attention \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --fold ${FOLD} \
    --dataset-name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --batch-size ${BATCH_SIZE} \
    ${EXTRA_ARGS}
else
  # MODE 2: pretrained checkpoint (explicit paths)
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Dataset: ${DATASET_NAME}"

  python -m med_slim.eval.xai.slice_attention \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --feat-dir "${FEAT_DIR}" \
    --annotations-path "${ANNOTATIONS_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --dataset-name "${DATASET_NAME}" \
    --plane "${PLANE}" \
    --fm-model-names "${FM_MODEL_NAMES}" \
    --split "${SPLIT}" \
    --batch-size ${BATCH_SIZE} \
    ${EXTRA_ARGS}
fi

echo "Slice attention visualization complete!"
