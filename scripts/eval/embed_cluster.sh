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
#SBATCH --account=p0021834    

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CHECKPOINT_PATH="/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-02-15-11:55/medslim-epoch2000.pth.tar"
FEAT_DIR="/hpcwork/rwth1833/feat_caches/MRNet/slices_raw/crop"
ANNOTATIONS_DIR="/hpcwork/rwth1833/datasets/preprocessed/MRNet"
OUTPUT_DIR="/hpcwork/rwth1833/experiments/MedSliM-linear-probing/embed_cluster"
DATASET_NAME="MRNet"
PLANE="sagittal"
FM_MODEL_NAMES="dinov2 medsiglip ark mri-core"

# Recommended future mode once experiment configs include cobra_config:
# EXPERIMENT_DIR="/hpcwork/rwth1833/experiments/MedSliM-linear-probing/meniscus_sagittal_2026-02-19-06:18"

# Visualization settings
METHOD="umap"           # Options: "umap", "tsne"
SLICE_LEVEL=true       # Set to true for slice-level embeddings
SUPERVISED=false        # Set to true for supervised UMAP (uses pathology labels to guide layout)

# UMAP hyperparameters
N_NEIGHBORS=30
MIN_DIST=0.05

# t-SNE hyperparameters
PERPLEXITY=30.0

### Build extra args
EXTRA_ARGS=""

if [[ "${SLICE_LEVEL}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --slice-level"
fi

if [[ "${SUPERVISED}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --supervised"
fi

### Run script
echo "Starting COBRA embedding extraction and ${METHOD} clustering..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Dataset: ${DATASET_NAME}"

python -m med_slim.eval.embed_cluster \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --feat-dir "${FEAT_DIR}" \
  --annotations-dir "${ANNOTATIONS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --plane "${PLANE}" \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --method "${METHOD}" \
  --n-neighbors $N_NEIGHBORS \
  --min-dist $MIN_DIST \
  --perplexity $PERPLEXITY \
  --save-embeddings \
  ${EXTRA_ARGS}

# Future experiment-dir mode:
# python -m med_slim.eval.embed_cluster \
#   --experiment-dir "${EXPERIMENT_DIR}" \
#   --method "${METHOD}" \
#   --save-embeddings \
#   ${EXTRA_ARGS}

echo "Embedding extraction and clustering complete!"
echo "Output saved to: ${OUTPUT_DIR}"
