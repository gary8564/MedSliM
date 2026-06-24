#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G 
#SBATCH --time=3:00:00                 
#SBATCH --job-name=precompute_slice_feature_%j
#SBATCH --output=stdout_precompute_slice_feature_%j.txt    
#SBATCH --account=p0021834


### Setup
source .venv/bin/activate
# Reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
DATA_DIR="/hpcwork/rwth1833/datasets/preprocessed/kneeMRI" #"/hpcwork/rwth1833/datasets/preprocessed/MRNet"
SAVE_DIR="/hpcwork/rwth1833/feat_caches/kneeMRI" #"/hpcwork/rwth1833/feat_caches/MRNet"
PLANE="sagittal"
USE_RAW_SLICE_RESOLUTION=true  
MODEL_NAME="curia" # "dinov2", "dinov3", "rad-dino", "medsiglip", "biomedclip", "ark", "mri-core", "medimageinsight", "curia"
SPLIT="test"
# Local checkpoints for models that require them
declare -A CHECKPOINTS=(
  ["ark"]="/hpcwork/rwth1833/models/Ark6_swinLarge768_ep50.pth.tar"
  ["mri-core"]="/hpcwork/rwth1833/models/mri_foundation.pth"
)
# Preprocessing modes: "resize", "resample", "crop", or "adaptive"
SPATIAL_MODE="crop"
# Tiled multi-crop CLS. 
# e.g., 0 = original global CLS; 4 = global + 2 x 2 regional crops.
REGIONAL_TOKENS=0
MRI_SEQUENCES="none"  # for fastMRI; use "none" when datasets do not contain multi-sequence
EXTRA_ARGS="--amp bf16"

# Conditionally extend extra args
if [ "$USE_RAW_SLICE_RESOLUTION" = true ]; then
  EXTRA_ARGS="$EXTRA_ARGS --use-raw-slice-resolution"
fi

if [ "$USE_RAW_SLICE_RESOLUTION" = false ]; then
  EXTRA_ARGS="$EXTRA_ARGS --num-slices 32"
fi

if [ "$REGIONAL_TOKENS" -gt 0 ]; then
  EXTRA_ARGS="$EXTRA_ARGS --regional-tokens $REGIONAL_TOKENS"
fi

if [ -n "${MRI_SEQUENCES:-}" ] && [ "$MRI_SEQUENCES" != "none" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --mri-sequences $MRI_SEQUENCES"
fi

# Add checkpoint for models that require local weights
if [[ -v CHECKPOINTS[$MODEL_NAME] ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --checkpoint ${CHECKPOINTS[$MODEL_NAME]} --workers 2"
fi

# Run your program
python ./med_slim/utils/preprocessing/precompute_slice_feature.py \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --plane "$PLANE" \
    --model-name "$MODEL_NAME" \
    --split "$SPLIT" \
    --spatial-mode "$SPATIAL_MODE" \
    $EXTRA_ARGS