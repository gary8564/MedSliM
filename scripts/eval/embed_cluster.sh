#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=1:00:00                 
#SBATCH --job-name=embed_cluster_%j
#SBATCH --output=stdout_embed_cluster_%j.txt    
#SBATCH --account=rwth1833    

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CHECKPOINT_PATH="/hpcwork/qj474765/checkpoints/MedSliM-pretraining/test-run-MRNet/2025-12-02-03:07/medslim_test_run_MRNet-epoch2000.pth.tar"
FEAT_DIR="/hpcwork/rwth1833/feat_caches/MRNet/slices_32"
ANNOTATIONS_PATH="/hpcwork/rwth1833/datasets/preprocessed/MRNet/test.csv"
OUTPUT_DIR="/hpcwork/rwth1833/experiments/MedSliM-linear-probing/embed_cluster/MRNet-2026-01-09-04:55"

# Model settings
FM_MODEL_NAMES="dinov2 rad-dino medsiglip ark biomedclip"
FM_POOLING="mean"  # Options: "mean", "concat"
SEQUENCE_ENCODER="mamba2"
SLICE_POOLING="abmil"

# Data settings
SPLIT="test"
PLANE="sagittal"
TARGET_LABELS="abnormal acl meniscus"  # Space-separated pathology labels

# Visualization settings
TITLE="MRNet Embedding Space"

### Run script
echo "Starting COBRA embedding extraction and UMAP clustering..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Target labels: ${TARGET_LABELS}"
echo "Plane: ${PLANE}, Split: ${SPLIT}"

python -m med_slim.eval.embed_cluster \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --feat-dir "${FEAT_DIR}" \
  --annotations-path "${ANNOTATIONS_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --plane "${PLANE}" \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --target-labels ${TARGET_LABELS} \
  --fm-pooling "${FM_POOLING}" \
  --sequence-encoder "${SEQUENCE_ENCODER}" \
  --slice-pooling "${SLICE_POOLING}" \
  --title "${TITLE}" \
  --save-embeddings

echo "Embedding extraction and clustering complete!"
echo "Output saved to: ${OUTPUT_DIR}"
