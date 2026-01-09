#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G 
#SBATCH --time=24:00:00                 
#SBATCH --job-name=pretrain_MRNet_sagittal
#SBATCH --output=stdout_pretrain_MRNet_sagittal_%j.txt    
#SBATCH --account=rwth1833    

### Setup
source .venv/bin/activate
# Reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


### Configuration
RESUME_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/test-run-MRNet/2026-01-09-04:57/medslim_test_run_MRNet-epoch600.pth.tar"
# PLANES="${PLANES:-sagittal coronal axial}"

### Run script
python ./med_slim/train/train.py \
  --resume "${RESUME_PATH}" \
  # --planes ${PLANES} \