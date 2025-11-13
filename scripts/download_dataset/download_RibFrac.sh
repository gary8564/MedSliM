#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RibFrac dataset configuration (Zenodo records)
# - Train Part 1: https://zenodo.org/records/3893508
# - Train Part 2: https://zenodo.org/records/3893498
# - Validation  : https://zenodo.org/records/3893496
# - Test        : https://zenodo.org/records/3993380

BASE_OUTPUT_DIR="/hpcwork/rwth1833/datasets/RibFrac"

declare -a RECORDS=(
  "3893508:train_part1"
  "3893498:train_part2"
  "3893496:val"
  "3993380:test"
)

log() { echo "[$SCRIPT_NAME] $*"; }
err() { echo "[$SCRIPT_NAME][ERROR] $*" >&2; }

print_usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME

Downloads the RibFrac dataset (all parts) from Zenodo using the helper
Python script 'fetch_from_zenoda.py'. Data will be organized under:
  $BASE_OUTPUT_DIR/{train_part1,train_part2,val,test}

Prerequisites:
  - python3 with 'requests' and 'python-dotenv' installed
  - Environment variable ZENODO_ACCESS_TOKEN set (or present in .env)

References:
  - Train Part 1: https://zenodo.org/records/3893508
  - Train Part 2: https://zenodo.org/records/3893498
  - Validation  : https://zenodo.org/records/3893496
  - Test        : https://zenodo.org/records/3993380
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

# Optional: basic dependency check
if ! python3 -c "import requests, dotenv" 2>/dev/null; then
  err "Missing Python deps. Install with: pip install requests python-dotenv"
  exit 1
fi

# Check token visibility (the python script also loads .env)
if [[ -z "${ZENODO_ACCESS_TOKEN:-}" ]]; then
  log "ZENODO_ACCESS_TOKEN not found in environment; relying on .env if present."
fi

log "Starting RibFrac dataset download..."
log "Base destination: $BASE_OUTPUT_DIR"

mkdir -p "$BASE_OUTPUT_DIR"

for entry in "${RECORDS[@]}"; do
  IFS=":" read -r RECORD_ID SUBDIR <<<"$entry"
  TARGET_DIR="$BASE_OUTPUT_DIR/$SUBDIR"
  mkdir -p "$TARGET_DIR"
  log "Downloading record $RECORD_ID -> $TARGET_DIR"
  if ! python3 "$PYTHON_SCRIPT" \
    --record_id "$RECORD_ID" \
    --output_folder "$TARGET_DIR"; then
    err "Failed to download Zenodo record $RECORD_ID"
    exit 1
  fi
done

# Simple summary
TOTAL_FILES=$(find "$BASE_OUTPUT_DIR" -type f | wc -l | awk '{print $1}')
TRAIN1_FILES=$(find "$BASE_OUTPUT_DIR/train_part1" -type f | wc -l | awk '{print $1}' || echo 0)
TRAIN2_FILES=$(find "$BASE_OUTPUT_DIR/train_part2" -type f | wc -l | awk '{print $1}' || echo 0)
VAL_FILES=$(find "$BASE_OUTPUT_DIR/val" -type f | wc -l | awk '{print $1}' || echo 0)
TEST_FILES=$(find "$BASE_OUTPUT_DIR/test" -type f | wc -l | awk '{print $1}' || echo 0)

log "Download completed successfully."
log "File counts: train_part1=$TRAIN1_FILES, train_part2=$TRAIN2_FILES, val=$VAL_FILES, test=$TEST_FILES, total=$TOTAL_FILES"
log "Data available under: $BASE_OUTPUT_DIR"


