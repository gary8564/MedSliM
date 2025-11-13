#!/bin/bash
#SBATCH --job-name=download_ctrate
#SBATCH --output=/home/qj474765/master_thesis/dima_3d/logs/download_ctrate_%j.out
#SBATCH --error=/home/qj474765/master_thesis/dima_3d/logs/download_ctrate_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Ensure log directory exists
mkdir -p /home/qj474765/master_thesis/dima_3d/logs

# Activate conda env
source ~/.bashrc
conda activate rad-dino

# Ensure we are in the project root
cd /home/qj474765/master_thesis/dima_3d

# User-configurable parameters via environment variables (override with --export)
DEST=${DEST:-/hpcwork/rwth1833/datasets/CT-RATE}
TRAIN_N=${TRAIN_N:-1500}
VALID_N=${VALID_N:-500}
EXT=${EXT:-.nii.gz}

echo "[INFO] Starting CT-RATE download job"
echo "[INFO] DEST=$DEST TRAIN_N=$TRAIN_N VALID_N=$VALID_N EXT=$EXT"

# Run the downloader
python /home/qj474765/master_thesis/dima_3d/scripts/download_dataset/download_CTRate.py \
  --dest "$DEST" \
  --train-n "$TRAIN_N" \
  --valid-n "$VALID_N" \
  --ext "$EXT"

echo "[OK] Job finished"