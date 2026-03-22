#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=2:00:00                 
#SBATCH --job-name=knn_classification_%j
#SBATCH --output=stdout_knn_classification_%j.txt    
#SBATCH --account=p0021834    

### Setup
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### Configuration
CONFIG_PATH="./med_slim/configs/linear_classifier.yml"
CHECKPOINT_PATH="/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-03-15-04:23/medslim-epoch2000.pth.tar"
SEQUENCE_ENCODER="mamba2"
SLICE_POOLING="cls"
FM_MODEL_NAMES="mri-core medimageinsight ark"
POOLING_TARGET="raw"
NB_KNN="10 20 50 100 200"
TEMPERATURE=0.07

EXTRA_ARGS=""
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --checkpoint-path ${CHECKPOINT_PATH}"
fi

if [[ "${SEQUENCE_ENCODER}" == "transformer" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --slice-pooling ${SLICE_POOLING}"
fi

### Run script
echo "Starting KNN evaluation..."

python ./med_slim/eval/knn_classifier.py \
  --config "${CONFIG_PATH}" \
  --fm-model-names "${FM_MODEL_NAMES}" \
  --sequence-encoder "${SEQUENCE_ENCODER}" \
  --pooling-target "${POOLING_TARGET}" \
  --nb-knn ${NB_KNN} \
  --temperature ${TEMPERATURE} \
  ${EXTRA_ARGS}

echo "KNN evaluation complete!"
