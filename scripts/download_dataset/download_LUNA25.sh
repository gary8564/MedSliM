#!/usr/bin/env bash
# 
# Script to download LUNA25 dataset (images and annotations) from Zenodo
# Usage: ./download_LUNA25.sh

set -euo pipefail

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LUNA25 dataset configuration
IMAGES_RECORD_ID="14223624"
ANNOTATIONS_RECORD_ID="14673658"
BASE_OUTPUT_DIR="/hpcwork/rwth1833/datasets/LUNA25"
IMAGES_OUTPUT_DIR="${BASE_OUTPUT_DIR}/images"
ANNOTATIONS_OUTPUT_DIR="${BASE_OUTPUT_DIR}/annot"

log() { echo "[$SCRIPT_NAME] $*"; }
err() { echo "[$SCRIPT_NAME][ERROR] $*" >&2; }

print_usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME

Downloads the LUNA25 dataset from Zenodo, including both images and annotations.

Dataset Information:
  - Images: 2120 patients, 4069 low-dose chest CT scans (~221.1 GB)
  - Annotations: 555 annotated malignant nodules and 5608 benign nodules
  - Images Record ID: $IMAGES_RECORD_ID
  - Annotations Record ID: $ANNOTATIONS_RECORD_ID

Output Structure:
  - Images: $IMAGES_OUTPUT_DIR
  - Annotations: $ANNOTATIONS_OUTPUT_DIR

USAGE
}

# Check for help
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

# Check if Python script exists
PYTHON_SCRIPT="${SCRIPT_DIR}/fetch_from_zenoda.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  err "Python script not found: $PYTHON_SCRIPT"
  exit 1
fi

# Check if Python is available
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 is not installed or not in PATH"
  exit 1
fi

# Check if requests module is available
if ! python3 -c "import requests" 2>/dev/null; then
  err "Python requests module is not installed. Please install it with: pip install requests"
  exit 1
fi

log "Starting LUNA25 dataset download..."
log "Base output directory: $BASE_OUTPUT_DIR"

# Download images
log "Downloading LUNA25 images (Record ID: $IMAGES_RECORD_ID)..."
log "Output directory: $IMAGES_OUTPUT_DIR"

if ! python3 "$PYTHON_SCRIPT" --record_id "$IMAGES_RECORD_ID" --output_folder "$IMAGES_OUTPUT_DIR"; then
  err "Failed to download images"
  exit 1
fi

log "Images download completed successfully!"

# Download annotations
log "Downloading LUNA25 annotations (Record ID: $ANNOTATIONS_RECORD_ID)..."
log "Output directory: $ANNOTATIONS_OUTPUT_DIR"

if ! python3 "$PYTHON_SCRIPT" --record_id "$ANNOTATIONS_RECORD_ID" --output_folder "$ANNOTATIONS_OUTPUT_DIR"; then
  err "Failed to download annotations"
  exit 1
fi

log "Annotations download completed successfully!"

# Summary
log "LUNA25 dataset download completed!"
log "Images saved to: $IMAGES_OUTPUT_DIR"
log "Annotations saved to: $ANNOTATIONS_OUTPUT_DIR"
log "Total dataset includes:"
log "  - 2120 patients"
log "  - 4069 low-dose chest CT scans"
log "  - 555 annotated malignant nodules"
log "  - 5608 benign nodules"
