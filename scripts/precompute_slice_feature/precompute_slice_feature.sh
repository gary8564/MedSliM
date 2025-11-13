#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=3:00:00                 
#SBATCH --job-name=precompute_slice_feature_ark_sagittal
#SBATCH --output=stdout_ark_sagittal.txt    
#SBATCH --account=rwth1833    


### Setup
source .venv/bin/activate
# Reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
DATA_DIR="/hpcwork/rwth1833/datasets/preprocessed/MRNet"
SAVE_DIR="/hpcwork/rwth1833/feat_caches/MRNet"
PLANE="sagittal"
NUM_SLICES=32
MODEL_NAME="ark"
ARK_PATH="/work/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar"
SPLIT="test"
BATCH_SIZE=8
EXTRA_ARGS="--amp"

# Conditionally extend extra args
if [[ "$MODEL_NAME" == "ark" ]]; then
  BATCH_SIZE=2
  EXTRA_ARGS+=" --ark-checkpoint $ARK_PATH"
fi

# Run your program
python scripts/precompute_slice_feature/precompute_slice_feature.py \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --plane "$PLANE" \
    --num-slices "$NUM_SLICES" \
    --model-name "$MODEL_NAME" \
    --split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    $EXTRA_ARGS