#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=preprocess_OAI
#SBATCH --output=stdout_preprocess_OAI_%j.txt

# CPU-only: DICOM tar.gz → NIfTI for OAI baseline MRI (SSL + thigh).
# Resume-safe; re-run the same command if the job times out.
# Submit from repo root:  sbatch scripts/preprocess_dataset/preprocess_OAI.sh

source .venv/bin/activate
python scripts/preprocess_dataset/preprocess_OAI.py \
  --data-dir /hpcwork/rwth1833/datasets/OAI \
  --save-dir /hpcwork/rwth1833/datasets/preprocessed/OAI \
  --bucket all \
  --workers 32
