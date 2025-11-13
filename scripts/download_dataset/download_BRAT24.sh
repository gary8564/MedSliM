#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# BRAT2024 dataset configuration
BRAT24_SYNAPSE_ID="syn64952546"
BASE_OUTPUT_DIR="/hpcwork/rwth1833/datasets/BRAT24"

log() { echo "[$SCRIPT_NAME] $*"; }
err() { echo "[$SCRIPT_NAME][ERROR] $*" >&2; }

print_usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME

Downloads the BRAT2024 dataset from Synapse.

Dataset Information:
  - Synapse Project/Folder ID: $BRAT24_SYNAPSE_ID
  - Source: BRAT2024 on Synapse

Output:
  - Download destination: $BASE_OUTPUT_DIR

Prerequisites:
  - python3 with 'synapseclient' and 'synapseutils' installed
  - Environment variable SYNAPSE_ACCESS_TOKEN set (or .env file)

The script will create the destination directory and download all files from the
Synapse project/folder into it.
USAGE
}

# Help
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

# Check Python helper exists
PYTHON_SCRIPT="${SCRIPT_DIR}/fetch_from_synapse.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  err "Python script not found: $PYTHON_SCRIPT"
  exit 1
fi

# Check python3
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 is not installed or not in PATH"
  exit 1
fi

# Check synapse packages
if ! python3 -c "import synapseclient, synapseutils" 2>/dev/null; then
  err "Missing Python deps. Install with: pip install synapseclient synapseutils python-dotenv"
  exit 1
fi

log "Starting BRAT2024 dataset download..."
log "Synapse ID: $BRAT24_SYNAPSE_ID"
log "Destination: $BASE_OUTPUT_DIR"

mkdir -p "$BASE_OUTPUT_DIR"

if ! python3 "$PYTHON_SCRIPT" \
  --synapse_id "$BRAT24_SYNAPSE_ID" \
  --output_dir "$BASE_OUTPUT_DIR"; then
  err "Failed to download BRAT2024 dataset from Synapse"
  exit 1
fi

FILE_COUNT=$(find "$BASE_OUTPUT_DIR" -type f | wc -l | awk '{print $1}')
DIR_COUNT=$(find "$BASE_OUTPUT_DIR" -type d | wc -l | awk '{print $1}')

log "Download completed successfully."
log "Items downloaded: $FILE_COUNT files across $((DIR_COUNT-1)) directories"
log "Data available at: $BASE_OUTPUT_DIR"


