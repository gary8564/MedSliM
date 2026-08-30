#!/usr/bin/bash
set -euo pipefail

### Job Parameters (optional SLURM; override DATA_DIR/SAVE_DIR on other machines)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=precompute_rsna_knee_%j
#SBATCH --output=logs/precompute/stdout_precompute_rsna_knee_%j.txt
#SBATCH --account=p0021834

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/precompute

### Configuration
DATA_DIR="${DATA_DIR:-/hpcwork/rwth1833/datasets/preprocessed/RSNA-Knee}"
SAVE_DIR="${SAVE_DIR:-/hpcwork/rwth1833/feat_caches/RSNA-Knee}"

# All view planes and splits present under the preprocessed dataset.
PLANES=(sagittal coronal axial)
SPLITS=(train test)

# All MRI sequence folders under {split}/ (fluid_sensitive_fs, non_fluid_sensitive).
MRI_SEQUENCES="all"

# Adaptive is preferred over crop for ~500x500 in-plane resolution:
# - 224-target FMs (dinov2/dinov3/biomedclip/mri-core): crop would discard most FOV;
#   adaptive downsamples after a limited intermediate crop.
# - ~448-518 targets (medsiglip/medimageinsight/rad-dino/curia): near 1:1 CropOrPad.
# - 768 ark: pads rather than inventing high-frequency detail via upsample.
SPATIAL_MODE="adaptive"
USE_RAW_SLICE_RESOLUTION=true
REGIONAL_TOKENS=0

MODEL_NAMES=(
  curia
  dinov2
  dinov3
  rad-dino
  medsiglip
  biomedclip
  ark
  mri-core
  medimageinsight
)

# Local checkpoints required by some extractors (HPC paths; override via env).
declare -A CHECKPOINTS=(
  ["ark"]="${ARK_CKPT:-/hpcwork/rwth1833/models/Ark6_swinLarge768_ep50.pth.tar}"
  ["mri-core"]="${MRI_CORE_CKPT:-/hpcwork/rwth1833/models/mri_foundation.pth}"
  ["medimageinsight"]="${MEDIMAGEINSIGHT_CKPT:-/hpcwork/rwth1833/models/MedImageInsights}"
)

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --save-dir "$SAVE_DIR"
  --spatial-mode "$SPATIAL_MODE"
  --mri-sequences "$MRI_SEQUENCES"
  --amp bf16
)

if [ "$USE_RAW_SLICE_RESOLUTION" = true ]; then
  COMMON_ARGS+=(--use-raw-slice-resolution)
else
  COMMON_ARGS+=(--num-slices 32)
fi

if [ "$REGIONAL_TOKENS" -gt 0 ]; then
  COMMON_ARGS+=(--regional-tokens "$REGIONAL_TOKENS")
fi

echo "============================================================"
echo "RSNA-Knee slice feature precompute"
echo "  data:       $DATA_DIR"
echo "  save:       $SAVE_DIR"
echo "  models:     ${MODEL_NAMES[*]}"
echo "  splits:     ${SPLITS[*]}"
echo "  planes:     ${PLANES[*]}"
echo "  sequences:  $MRI_SEQUENCES"
echo "  spatial:    $SPATIAL_MODE"
echo "  raw slices: $USE_RAW_SLICE_RESOLUTION"
echo "  regional:   $REGIONAL_TOKENS"
echo "============================================================"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  EXTRA_ARGS=()

  if [[ -v CHECKPOINTS[$MODEL_NAME] ]]; then
    CKPT="${CHECKPOINTS[$MODEL_NAME]}"
    if [ ! -e "$CKPT" ]; then
      echo "ERROR: missing checkpoint for ${MODEL_NAME}: ${CKPT}" >&2
      echo "Skip ${MODEL_NAME}, or place/clone the weights and re-run." >&2
      continue
    fi
    EXTRA_ARGS+=(--checkpoint "$CKPT" --workers 2)
  fi

  # Curia intensity preprocessing is modality-aware.
  if [ "$MODEL_NAME" = "curia" ]; then
    EXTRA_ARGS+=(--modality mri)
  fi

  for SPLIT in "${SPLITS[@]}"; do
    for PLANE in "${PLANES[@]}"; do
      echo "------------------------------------------------------------"
      echo "Running model=${MODEL_NAME} split=${SPLIT} plane=${PLANE}"
      echo "------------------------------------------------------------"
      python ./med_slim/utils/preprocessing/precompute_slice_feature.py \
        "${COMMON_ARGS[@]}" \
        --model-name "$MODEL_NAME" \
        --split "$SPLIT" \
        --plane "$PLANE" \
        "${EXTRA_ARGS[@]}"
    done
  done
done

echo "============================================================"
echo "RSNA-Knee precompute finished."
echo "============================================================"
