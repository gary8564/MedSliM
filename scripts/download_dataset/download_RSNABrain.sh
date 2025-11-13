#!/bin/bash
#SBATCH --job-name=download_rsnabrain
#SBATCH --output=/home/qj474765/master_thesis/dima_3d/logs/download_rsnabrain_%j.out
#SBATCH --error=/home/qj474765/master_thesis/dima_3d/logs/download_rsnabrain_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Activate conda env
source ~/.bashrc
conda activate rad-dino

mkdir -p /hpcwork/rwth1833/datasets/RSNA_Brain_Tumor_Radiogenomic/
kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification -p /hpcwork/rwth1833/datasets/RSNA_Brain_Tumor_Radiogenomic/