#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G 
#SBATCH --time=36:00:00                 
#SBATCH --job-name=precompute_slice_feature_%j
#SBATCH --output=stdout_precompute_slice_feature_%j.txt    
#SBATCH --account=rwth1833    


### Setup
source .venv/bin/activate
# Reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
DATA_DIR="/work/rwth1833/datasets/preprocessed/fastMRI"
SAVE_DIR="/hpcwork/rwth1833/feat_caches/fastMRI"
PLANE="axial"
USE_RAW_SLICE_RESOLUTION=false
MODEL_NAME="biomedclip" # "dinov2", "dinov3", "rad-dino", "medsiglip", "biomedclip", "ark"
SPLIT="train"
ARK_CHECKPOINT="/work/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar"
EXTRA_ARGS="--amp bf16"

# Conditionally extend extra args
if [ "$USE_RAW_SLICE_RESOLUTION" = true ]; then
  EXTRA_ARGS="$EXTRA_ARGS --use-raw-slice-resolution"
fi

if [ "$USE_RAW_SLICE_RESOLUTION" = false ]; then
  EXTRA_ARGS="$EXTRA_ARGS --num-slices 32"
fi

# Add ark checkpoint if using ark model
if [ "$MODEL_NAME" = "ark" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --ark-checkpoint $ARK_CHECKPOINT --workers 2"
fi

# Run your program
python ./med_slim/utils/preprocessing/precompute_slice_feature.py \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --plane "$PLANE" \
    --model-name "$MODEL_NAME" \
    --split "$SPLIT" \
    $EXTRA_ARGS