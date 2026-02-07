#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G 
#SBATCH --time=6:00:00                 
#SBATCH --job-name=pretrain_MRNet
#SBATCH --output=logs/pretrain/stdout_pretrain_MRNet_%j.txt    
#SBATCH --account=rwth1833    

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
# RESUME_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/MRNet-fastMRI/2026-01-31-16:40/medslim-epoch300.pth.tar"
# PLANES="sagittal coronal axial"

### Run script with Accelerate for multi-GPU training
# Use bf16 instead of fp16 for numerical stability
# Available options:
#   --sequence-encoder mamba2/transformer
#   --pooling abmil          # abmil (default) or cls (requires transformer encoder)
#   --resume "${RESUME_PATH}"              # Continue training from checkpoint (keeps the state of optimizer and epoch)
#   --resume "${RESUME_PATH}" --curriculum # Curriculum learning: load weights only, reset optimizer and epoch
#   --planes ${PLANES}

accelerate launch --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    ./med_slim/train/train.py