#!/usr/bin/bash
# Few-shot label-efficiency sweep.
# TRAIN_FRACTION / NUM_REPEATS must be passed via sbatch --export; otherwise
# the batch job only sees defaults from linear_classifier.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."

NUM_REPEATS=1
for frac in 0.10 0.25 0.5 0.75; do
  sbatch --export=ALL,TRAIN_FRACTION=${frac},NUM_REPEATS=${NUM_REPEATS} \
    scripts/eval/linear_classifier.sh
done
