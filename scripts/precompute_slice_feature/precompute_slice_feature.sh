#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G 
#SBATCH --time=1:00:00                 
#SBATCH --job-name=precompute_slice_feature_%j
#SBATCH --output=stdout_precompute_slice_feature_%j.txt    
#SBATCH --account=rwth1833    


### Setup
source .venv/bin/activate
# Reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
DATA_DIR="/work/rwth1833/datasets/preprocessed/fastMRI" #"/hpcwork/rwth1833/datasets/preprocessed/MRNet"
SAVE_DIR="/hpcwork/rwth1833/feat_caches/fastMRI" #"/hpcwork/rwth1833/feat_caches/MRNet"
PLANE="axial"
USE_RAW_SLICE_RESOLUTION=false  
MODEL_NAME="medsiglip" # "dinov2", "dinov3", "rad-dino", "medsiglip", "biomedclip", "ark", "mri-core"
SPLIT="train"
# Local checkpoints for models that require them
declare -A CHECKPOINTS=(
  ["ark"]="/work/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar"
  ["mri-core"]="/work/rwth1833/models/mri_core/mri_foundation.pth"
)
# Preprocessing modes: "resize", "resample", "crop", or "adaptive"
SPATIAL_MODE="adaptive"
EXTRA_ARGS="--amp bf16"

# Conditionally extend extra args
if [ "$USE_RAW_SLICE_RESOLUTION" = true ]; then
  EXTRA_ARGS="$EXTRA_ARGS --use-raw-slice-resolution"
fi

if [ "$USE_RAW_SLICE_RESOLUTION" = false ]; then
  EXTRA_ARGS="$EXTRA_ARGS --num-slices 32"
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