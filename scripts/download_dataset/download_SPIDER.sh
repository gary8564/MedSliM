#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SPIDER dataset (Zenodo) configuration
# Record page: https://zenodo.org/records/10159290
SPIDER_RECORD_ID="10159290"
BASE_OUTPUT_DIR="/hpcwork/rwth1833/datasets/SPIDER"

log() { echo "[$SCRIPT_NAME] $*"; }
err() { echo "[$SCRIPT_NAME][ERROR] $*" >&2; }

print_usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME

Downloads the SPIDER lumbar spine MRI dataset from Zenodo using the helper
Python script 'fetch_from_zenoda.py'. All files from the Zenodo record will
be saved into:
  $BASE_OUTPUT_DIR

Prerequisites:
  - python3 with 'requests' and 'python-dotenv' installed
  - Environment variable ZENODO_ACCESS_TOKEN set (or present in .env)

Reference:
  - SPIDER dataset (Zenodo): https://zenodo.org/records/10159290
USAGE
}

# Help
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

# Check Python helper exists
PYTHON_SCRIPT="${SCRIPT_DIR}/fetch_from_zenoda.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  err "Python script not found: $PYTHON_SCRIPT"
  exit 1
fi

# Check python3
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 is not installed or not in PATH"
  exit 1
fi

# Dependency check
if ! python3 -c "import requests, dotenv" 2>/dev/null; then
  err "Missing Python deps. Install with: pip install requests python-dotenv"
  exit 1
fi

# Token notice (optional; python script also loads from .env)
if [[ -z "${ZENODO_ACCESS_TOKEN:-}" ]]; then
  log "ZENODO_ACCESS_TOKEN not found in environment; relying on .env if present."
fi

log "Starting SPIDER dataset download..."
log "Record ID: $SPIDER_RECORD_ID"
log "Destination: $BASE_OUTPUT_DIR"

mkdir -p "$BASE_OUTPUT_DIR"

if ! python3 "$PYTHON_SCRIPT" \
  --record_id "$SPIDER_RECORD_ID" \
  --output_folder "$BASE_OUTPUT_DIR"; then
  err "Failed to download SPIDER dataset from Zenodo"
  exit 1
fi

# Summary
FILE_COUNT=$(find "$BASE_OUTPUT_DIR" -type f | wc -l | awk '{print $1}')
DIR_COUNT=$(find "$BASE_OUTPUT_DIR" -type d | wc -l | awk '{print $1}')

log "Download completed successfully."
log "Items downloaded: $FILE_COUNT files across $((DIR_COUNT-1)) directories"
log "Data available at: $BASE_OUTPUT_DIR"


