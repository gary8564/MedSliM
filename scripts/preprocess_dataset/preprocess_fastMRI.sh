#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=preprocess_fastMRI_train
#SBATCH --output=stdout_preprocess_fastMRI_train_%j.txt

source .venv/bin/activate
python scripts/preprocess_dataset/preprocess_fastMRI.py --split train --workers 32