#!/usr/bin/env bash
set -euo pipefail

# Usage: ./download_NLST.sh [OUTPUT_DIR]

RECORD_ID="14838349" # NLSTseg: https://zenodo.org/records/14838349
DEST="${1:-/hpcwork/rwth1833/datasets/NLST/NLSTseg}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_FETCH="${SCRIPT_DIR}/fetch_from_zenoda.py"

mkdir -p "$DEST"

echo "[NLSTseg] Downloading record ${RECORD_ID} to: $DEST"
python3 "$PY_FETCH" --record_id "$RECORD_ID" --output_folder "$DEST"
echo "[NLSTseg] Done. Data at: $DEST"


