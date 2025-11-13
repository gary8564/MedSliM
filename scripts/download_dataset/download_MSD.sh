#!/usr/bin/env bash
set -euo pipefail

DEST="/hpcwork/rwth1833/datasets/MSD"
mkdir -p "$DEST"
cd "$DEST"

# Prefer EU (London) mirror from Registry of Open Data on AWS; fall back to US-West-2
EU_BASE="https://msd-for-monai-eu.s3.eu-west-2.amazonaws.com"
US_BASE="https://msd-for-monai.s3-us-west-2.amazonaws.com"

# Choose the tasks you want:
TASKS=(
  Task01_BrainTumour
  Task02_Heart
  Task03_Liver
  Task04_Hippocampus
  Task05_Prostate
  Task06_Lung
  Task07_Pancreas
  Task08_HepaticVessel
  Task09_Spleen
  Task10_Colon
)

# Use aria2c if available (better resume/parallel), else wget -c (resume)
dl() {
  local url="$1" out="$2"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x8 -s8 -o "$out" "$url" || return 1
  else
    wget -c -O "$out" "$url" || return 1
  fi
}

for t in "${TASKS[@]}"; do
  tarfile="${t}.tar"
  # Skip if already extracted
  if [ -d "$t" ]; then
    echo "[skip] $t already extracted"
    continue
  fi
  # Skip if fully downloaded
  if [ -f "$tarfile" ]; then
    echo "[resume] $tarfile exists; will resume/verify"
  fi
  echo "[download] $tarfile (EU)"
  if ! dl "${EU_BASE}/${tarfile}" "$tarfile"; then
    echo "[retry] $tarfile from US mirror"
    dl "${US_BASE}/${tarfile}" "$tarfile"
  fi

  echo "[extract] $tarfile"
  tar -xf "$tarfile"
done

echo "All done. Data lives in: $DEST"
