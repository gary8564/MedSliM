#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=2:00:00                 
#SBATCH --job-name=roi_occlusion_%j
#SBATCH --output=logs/eval/stdout_roi_occlusion_%j.txt
#SBATCH --account=p0021834    

### Setup
set -e
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
# Occlusion scores the shipped classifier (COBRA + MLP head), so it always loads
# fold_*/ckpt/classifier.pt from a linear-probing experiment. There is no
# pretrained-checkpoint mode: an SSL checkpoint has no head and therefore no logits.
# Same experiment-dir convention as scripts/eval/slice_attention.sh MODE 1.
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/hpcwork/rwth1833/experiments/MedSliM-linear-probing/meniscal_tear_ligament_tear_cartilage_lesion_effusion_DESS_E2/sagittal_2026-03-17-01:46_65920427}"
FOLD=3
DATASET_NAME="${DATASET_NAME:-SKM-TEA}"

# Occlusion settings
SPLIT="test"
NUM_SAMPLES=""          # Leave empty for all samples, or set number (e.g., "20")
N_RANDOM=5              # Control windows averaged per volume in Experiment A
SEED=0
SKIP_LOSO=false         # Skip Experiment B (also drops the occlusion MoRF ranking)
SKIP_MORF=false         # Skip Experiment C

### Build extra args
EXTRA_ARGS=""

if [[ -n "${NUM_SAMPLES}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --num-samples ${NUM_SAMPLES}"
fi

if [[ "${SKIP_LOSO}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --skip-loso"
fi

if [[ "${SKIP_MORF}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --skip-morf"
fi

### Run script
echo "Starting ROI occlusion explanations ..."
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Fold: ${FOLD}"
echo "Dataset: ${DATASET_NAME}"

# --batch-size stays 1: every drop shortens the bag by a different amount.
python -m med_slim.eval.xai.roi_occlusion \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --fold ${FOLD} \
  --dataset-name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --batch-size 1 \
  --n-random ${N_RANDOM} \
  --seed ${SEED} \
  ${EXTRA_ARGS}

echo "ROI occlusion explanations complete!"
