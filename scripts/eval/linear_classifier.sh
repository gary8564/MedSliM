#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=4:00:00                 
#SBATCH --job-name=lp_MRNet_binary_classification
#SBATCH --output=stdout_lp_MRNet_binary_classification.txt    
#SBATCH --account=rwth1833    

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
PRETRAIN_CONFIG_PATH="${PRETRAIN_CONFIG_PATH:-./med_slim/configs/pretrain.yaml}"
CONFIG_PATH="${CONFIG_PATH:-./med_slim/configs/linear_classifier.yml}"

### Run script
echo "Starting linear classifier evaluation..."

python ./med_slim/eval/linear_classifier.py \
  --pretrain-config "${PRETRAIN_CONFIG_PATH}" \
  --linear-classifier-config "${CONFIG_PATH}" \
  --deploy

echo "Linear probing evaluation complete!"

