#!/usr/bin/env bash
# 
# Script to download BTCV (Beyond the Cranial Vault) dataset from Synapse
# Usage: ./download_BTCV.sh

set -euo pipefail

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# BTCV dataset configuration
TRAINING_IMAGES_SYNAPSE_ID="syn10285054"
TRAINING_LABELS_SYNAPSE_ID="syn10285076"
BASE_OUTPUT_DIR="/hpcwork/rwth1833/datasets/BTCV"
IMAGES_OUTPUT_DIR="${BASE_OUTPUT_DIR}/images"
ANNOTATIONS_OUTPUT_DIR="${BASE_OUTPUT_DIR}/annot"

log() { echo "[$SCRIPT_NAME] $*"; }
err() { echo "[$SCRIPT_NAME][ERROR] $*" >&2; }

print_usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME

Downloads the BTCV (Beyond the Cranial Vault) training dataset from Synapse.

Dataset Information:
  - Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge
  - Training Images Synapse ID: $TRAINING_IMAGES_SYNAPSE_ID
  - Training Labels Synapse ID: $TRAINING_LABELS_SYNAPSE_ID
  - Contains abdominal CT training images and corresponding organ segmentations
  - Used for multi-organ segmentation challenges

Output Structure:
  - Images: $IMAGES_OUTPUT_DIR
  - Annotations: $ANNOTATIONS_OUTPUT_DIR

Prerequisites:
  - synapseclient Python package installed
  - SYNAPSE_ACCESS_TOKEN environment variable set
  - .env file with SYNAPSE_ACCESS_TOKEN (optional)

The script will create the necessary directories and download training files.
USAGE
}

# Check for help
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

# Check if Python script exists
PYTHON_SCRIPT="${SCRIPT_DIR}/fetch_from_synapse.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  err "Python script not found: $PYTHON_SCRIPT"
  exit 1
fi

# Check if Python is available
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 is not installed or not in PATH"
  exit 1
fi

# Check if synapseclient is available
if ! python3 -c "import synapseclient, synapseutils" 2>/dev/null; then
  err "synapseclient package is not installed. Please install it with:"
  err "  pip install synapseclient"
  exit 1
fi

log "Starting BTCV training dataset download..."
log "Training Images Synapse ID: $TRAINING_IMAGES_SYNAPSE_ID"
log "Training Labels Synapse ID: $TRAINING_LABELS_SYNAPSE_ID"
log "Base output directory: $BASE_OUTPUT_DIR"

# Create target directories
mkdir -p "$IMAGES_OUTPUT_DIR"
mkdir -p "$ANNOTATIONS_OUTPUT_DIR"

# Download training images
log "Downloading training images..."
if ! python3 "$PYTHON_SCRIPT" --synapse_id "$TRAINING_IMAGES_SYNAPSE_ID" --output_dir "$IMAGES_OUTPUT_DIR"; then
  err "Failed to download training images from Synapse"
  exit 1
fi
log "Training images download completed successfully!"

# Download training labels/annotations
log "Downloading training labels..."
if ! python3 "$PYTHON_SCRIPT" --synapse_id "$TRAINING_LABELS_SYNAPSE_ID" --output_dir "$ANNOTATIONS_OUTPUT_DIR"; then
  err "Failed to download training labels from Synapse"
  exit 1
fi
log "Training labels download completed successfully!"

# Summary
log "BTCV training dataset download completed!"
log "Training images saved to: $IMAGES_OUTPUT_DIR"
log "Training labels saved to: $ANNOTATIONS_OUTPUT_DIR"

# Count files
IMAGE_COUNT=$(find "$IMAGES_OUTPUT_DIR" -type f | wc -l)
ANNOTATION_COUNT=$(find "$ANNOTATIONS_OUTPUT_DIR" -type f | wc -l)

log "File counts:"
log "  - Training Images: $IMAGE_COUNT files"
log "  - Training Labels: $ANNOTATION_COUNT files"

log "Download completed successfully!"
log "Both training images and labels are now organized in separate directories"
