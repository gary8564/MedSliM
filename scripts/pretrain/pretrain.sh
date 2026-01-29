#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G 
#SBATCH --time=72:00:00                 
#SBATCH --job-name=pretrain_MRNet_fastMRI
#SBATCH --output=stdout_pretrain_MRNet_fastMRI_%j.txt    
#SBATCH --account=p0021834    

### Setup
source .venv/bin/activate

# Reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Debug: Uncomment to see which parameters don't receive gradients in DDP
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Multi-GPU settings
NUM_GPUS=2

### Configuration
# RESUME_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/test-run-MRNet/2026-01-09-04:57/medslim_test_run_MRNet-epoch600.pth.tar"
# PLANES="sagittal coronal axial"

### Run script with Accelerate for multi-GPU training
# Use bf16 instead of fp16 for numerical stability
accelerate launch --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    ./med_slim/train/train.py
    # Optional args:
    # --sequence-encoder transformer 
    # --pooling cls 
    # --resume "${RESUME_PATH}" 
    # --planes ${PLANES} 