#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00                 
#SBATCH --job-name=medslim_test_run_MRNet
#SBATCH --output=logs/pretrain/stdout_pretrain_linspace_balanced_sampling_%j.txt    
#SBATCH --account=p0021834    

### Setup
# Load Intel libraries (required by Triton for mamba_ssm kernels)
module load intel 2>/dev/null || true

source .venv/bin/activate

# Reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Debug: Uncomment to see which parameters don't receive gradients in DDP
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Multi-GPU settings
NUM_GPUS=1

### Configuration
# Checkpoint to resume from (leave empty for training from scratch)
#RESUME_PATH="/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/test-run-MRNet/2026-02-08-18:35/medslim-epoch2000.pth.tar"

# Curriculum learning: load model weights only, reset optimizer and epoch.
# Set to true when adding new datasets or adding new slice encoder models.
#CURRICULUM=true

# Override slice encoder models from config
# Available: dinov2, dinov3, rad-dino, medsiglip, biomedclip, ark, mri-core
MODEL_NAMES="dinov2 dinov3 rad-dino medsiglip biomedclip ark mri-core"

# Override view planes from config (space-separated, leave empty to use config defaults)
# Available: axial, sagittal, coronal
PLANES=""

# Sequence encoder and pooling
SEQUENCE_ENCODER="mamba2"   # mamba2 or transformer
POOLING="abmil"             # abmil (default) or cls (requires transformer encoder)

### Build command arguments
EXTRA_ARGS=""
[[ -n "${RESUME_PATH}" ]]  && EXTRA_ARGS+=" --resume ${RESUME_PATH}"
[[ "${CURRICULUM}" == true ]] && EXTRA_ARGS+=" --curriculum"
[[ -n "${MODEL_NAMES}" ]]  && EXTRA_ARGS+=" --model-names ${MODEL_NAMES}"
[[ -n "${PLANES}" ]]       && EXTRA_ARGS+=" --planes ${PLANES}"

### Run
accelerate launch --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    ./med_slim/train/train.py \
    --sequence-encoder "${SEQUENCE_ENCODER}" \
    --pooling "${POOLING}" \
    ${EXTRA_ARGS}