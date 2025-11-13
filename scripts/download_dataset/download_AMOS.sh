#!/usr/bin/env bash
# 
# Simple script to download the AMOS dataset from Zenodo
# Usage: ./download_AMOS.sh [OUTPUT_DIR]

set -euo pipefail

SCRIPT_NAME=$(basename "$0")
OUTPUT_DIR="${1:-/hpcwork/rwth1833/datasets/AMOS}"

# AMOS dataset details
DOWNLOAD_URL="https://zenodo.org/records/7155725/files/amos22.zip?download=1"
FILENAME="amos22.zip"
EXPECTED_MD5="67717b2a483ac0744c89c3016b7aaef7"
SIZE="24.2 GB"

log() { echo "[$SCRIPT_NAME] $*"; }
err() { echo "[$SCRIPT_NAME][ERROR] $*" >&2; }

print_usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME [OUTPUT_DIR]

Downloads the AMOS (Abdominal Multi-Organ Segmentation) dataset from Zenodo.

AMOS is a large-scale, diverse, clinical dataset for abdominal organ segmentation.
It provides 500 CT and 100 MRI scans with voxel-level annotations of 15 abdominal organs.

Arguments:
  OUTPUT_DIR    Output directory (default: $OUTPUT_DIR)

Examples:
  $SCRIPT_NAME
  $SCRIPT_NAME /path/to/custom/directory

Dataset Information:
  - DOI: 10.5281/zenodo.7155725
  - Size: $SIZE
  - Format: 500 CT scans + 100 MRI scans
  - Annotations: 15 abdominal organs
  - License: CC BY 4.0
USAGE
}

# Check for help
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

# Check for download tools
if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
  err "Neither wget nor curl is installed. Please install one of them."
  exit 1
fi

# Check for checksum tool
if ! command -v md5sum >/dev/null 2>&1; then
  log "Warning: md5sum not available, skipping checksum verification"
  SKIP_CHECKSUM=true
else
  SKIP_CHECKSUM=false
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"

log "Downloading AMOS dataset to: $OUTPUT_FILE"
log "Dataset size: $SIZE"

# Download the file
if command -v wget >/dev/null 2>&1; then
  log "Using wget to download..."
  wget -c -O "$OUTPUT_FILE" "$DOWNLOAD_URL"
else
  log "Using curl to download..."
  curl -fL --retry 5 --retry-delay 5 -C - -o "$OUTPUT_FILE" "$DOWNLOAD_URL"
fi

# Verify checksum if possible
if [[ "$SKIP_CHECKSUM" == false ]]; then
  log "Verifying checksum..."
  ACTUAL_MD5=$(md5sum "$OUTPUT_FILE" | awk '{print $1}')
  if [[ "$ACTUAL_MD5" == "$EXPECTED_MD5" ]]; then
    log "Checksum verification successful"
  else
    err "Checksum verification failed!"
    err "Expected: $EXPECTED_MD5"
    err "Actual:   $ACTUAL_MD5"
    exit 1
  fi
fi

log "Download completed successfully!"
log "File saved as: $OUTPUT_FILE"
log "Dataset includes 500 CT and 100 MRI scans with 15 abdominal organ annotations"
