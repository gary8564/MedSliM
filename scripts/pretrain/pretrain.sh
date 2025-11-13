#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G 
#SBATCH --time=24:00:00                 
#SBATCH --job-name=pretrain_MRNet
#SBATCH --output=stdout_pretrain_MRNet.txt    
#SBATCH --account=rwth1833    

### Setup
source .venv/bin/activate
# Reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


### Configuration
RESUME_PATH="${RESUME_PATH:-}"


### Run script
python ./med_slim/train/train.py \
  ${RESUME_PATH:+--resume "${RESUME_PATH}"}