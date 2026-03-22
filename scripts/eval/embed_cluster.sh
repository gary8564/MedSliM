#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=00:30:00                 
#SBATCH --job-name=embed_cluster_%j
#SBATCH --output=stdout_embed_cluster_%j.txt    
#SBATCH --account=p0021834    

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Mode 1: Multi-dataset (cross-dataset, color by plane)
CHECKPOINT_PATH="/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-03-15-04:23/medslim-epoch2000.pth.tar"
OUTPUT_DIR="/hpcwork/rwth1833/experiments/MedSliM-linear-probing/embed_cluster_cross_dataset"
FM_MODEL_NAMES="mri-core medimageinsight ark"
# Visualization settings
METHOD="umap"
#MAX_SAMPLES_PER_GROUP=750
# UMAP hyperparameters
N_NEIGHBORS=175
MIN_DIST=0.1
# Preprocessing
PCA_DIM=100      # set to 0 to disable PCA denoising
# t-SNE hyperparameters
PERPLEXITY=30.0
# Pooling target
POOLING_TARGET="post_embed"
# Extra arguments
EXTRA_ARGS=""
[[ -n "${MAX_SAMPLES_PER_GROUP}" ]] && EXTRA_ARGS="${EXTRA_ARGS} --max-samples-per-group ${MAX_SAMPLES_PER_GROUP}"

echo "Starting cross-dataset COBRA embedding visualization ..."
echo "Checkpoint: ${CHECKPOINT_PATH}"

python -m med_slim.eval.embed_cluster \
  --multi-dataset \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --output-dir "${OUTPUT_DIR}" \
  --method "${METHOD}" \
  --n-neighbors $N_NEIGHBORS \
  --min-dist $MIN_DIST \
  --pca-dim $PCA_DIM \
  --pooling-target "${POOLING_TARGET}" \
  --save-embeddings \
  ${EXTRA_ARGS}

### Mode 2: Single-dataset (color by pathology)

CHECKPOINT_PATH="/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-03-15-04:23/medslim-epoch2000.pth.tar"
FEAT_DIR="/hpcwork/rwth1833/feat_caches/MRNet/slices_raw/crop"
ANNOTATIONS_DIR="/hpcwork/rwth1833/datasets/preprocessed/MRNet"
OUTPUT_DIR="/hpcwork/rwth1833/experiments/MedSliM-linear-probing/embed_cluster"
DATASET_NAME="MRNet"
PLANE="sagittal"
FM_MODEL_NAMES="mri-core medimageinsight ark"
TARGET_LABELS="abnormal"       # CSV column to color by (abnormal, acl, meniscus)
TASK="binary"
METHOD="umap"
POOLING_TARGET="raw"
SUPERVISED=false
N_NEIGHBORS=10
MIN_DIST=0.01
PERPLEXITY=30.0
SLICE_LEVEL=true

EXTRA_ARGS=""
[[ "${SLICE_LEVEL}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --slice-level"
[[ "${SUPERVISED}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --supervised"

python -m med_slim.eval.embed_cluster \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --feat-dir "${FEAT_DIR}" \
  --annotations-dir "${ANNOTATIONS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --plane "${PLANE}" \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --target-labels ${TARGET_LABELS} \
  --task "${TASK}" \
  --method "${METHOD}" \
  --pooling-target "${POOLING_TARGET}" \
  --n-neighbors $N_NEIGHBORS \
  --min-dist $MIN_DIST \
  --perplexity $PERPLEXITY \
  --save-embeddings \
  ${EXTRA_ARGS}

## Or use experiment-dir (single-dataset, auto-resolves model + config)
# EXPERIMENT_DIR="/hpcwork/rwth1833/experiments/MedSliM-linear-probing/acl_sagittal_2026-03-09-01:15"
# SUPERVISED=false
# SLICE_LEVEL=true
#
# EXTRA_ARGS=""
# [[ "${SLICE_LEVEL}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --slice-level"
# [[ "${SUPERVISED}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --supervised"
#
# python -m med_slim.eval.embed_cluster \
#   --experiment-dir "${EXPERIMENT_DIR}" \
#   --method "${METHOD}" \
#   --n-neighbors $N_NEIGHBORS \
#   --min-dist $MIN_DIST \
#   --perplexity $PERPLEXITY \
#   --save-embeddings \
#   ${EXTRA_ARGS}

echo "Embedding extraction and clustering complete!"
echo "Output saved to: ${OUTPUT_DIR}"
