#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=3:00:00                 
#SBATCH --job-name=precompute_slice_feature_rad-dino_axial_train
#SBATCH --output=stdout_rad-dino_axial_train.txt    
#SBATCH --account=rwth1833    


### Setup
source .venv/bin/activate
# Reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
DATA_DIR="/hpcwork/rwth1833/datasets/preprocessed/MRNet"
SAVE_DIR="/hpcwork/rwth1833/feat_caches/MRNet"
PLANE="axial"
USE_RAW_SLICE_RESOLUTION=true
MODEL_NAME="rad-dino"
SPLIT="train"
EXTRA_ARGS="--amp"

# Conditionally extend extra args
if [ "$USE_RAW_SLICE_RESOLUTION" = true ]; then
  EXTRA_ARGS="$EXTRA_ARGS --use-raw-slice-resolution"
fi

if [ "$USE_RAW_SLICE_RESOLUTION" = false ]; then
  EXTRA_ARGS="$EXTRA_ARGS --num-slices 32"
fi

# Run your program
python ./med_slim/utils/preprocessing/precompute_slice_feature.py \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --plane "$PLANE" \
    --model-name "$MODEL_NAME" \
    --split "$SPLIT" \
    $EXTRA_ARGS