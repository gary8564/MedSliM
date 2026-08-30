#!/usr/bin/bash
# KneeMRI Curia classifier token ablations (official SGD recipe).
#
# Runs the MedSliM-matched recipes (full FOV, no ACL bbox crop) against one
# shared token cache. The first recipe builds the cache; later recipes reuse it.
#
# Usage (from repo root):
#   bash scripts/eval/curia_classifier_ablation.sh
#   RECIPES="fullstack_cls cls_patch_mix" bash scripts/eval/curia_classifier_ablation.sh
#   sbatch scripts/eval/curia_classifier_ablation.sh

### Job Parameters
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=72:00:00
#SBATCH --job-name=curia_ablation_%j
#SBATCH --output=logs/eval/stdout_curia_ablation_%j.txt
#SBATCH --account=p0021834

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Curia weights are gated (RAIL-M). Accept the license at
# https://huggingface.co/raidium/curia and export HF_TOKEN before submitting.

BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/baselines/curia/configs/kneeMRI.yml}"
ABLATION_DIR="${ABLATION_DIR:-${ROOT_DIR}/baselines/curia/configs/ablations}"
DATA_DIR="${DATA_DIR:-/hpcwork/rwth1833/datasets/preprocessed/kneeMRI}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-/hpcwork/rwth1833/datasets/preprocessed/kneeMRI}"
OUTPUT_DIR="${OUTPUT_DIR:-/hpcwork/rwth1833/experiments/curia-classifier}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-/hpcwork/rwth1833/feat_caches/curia_official}"
WEIGHTED_LOSS="${WEIGHTED_LOSS:-true}"
N_FOLDS="${N_FOLDS:-3}"
SEED="${SEED:-42}"

# (1) full-stack patch means  (2) full-stack CLS  (3) CLS + patch mix
ALL_RECIPES=(fullstack_patch_mean fullstack_cls cls_patch_mix)
if [[ -n "${RECIPES:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_RECIPES=(${RECIPES})
else
  RUN_RECIPES=("${ALL_RECIPES[@]}")
fi

EXTRA_ARGS=""
if [[ "${WEIGHTED_LOSS}" == "true" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --weighted-loss"
fi

merge_config() {
  local base="$1"
  local overlay="$2"
  local dest="$3"
  python - "${base}" "${overlay}" "${dest}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path, overlay_path, dest_path = sys.argv[1], sys.argv[2], sys.argv[3]


def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value


with open(base_path) as handle:
    cfg = yaml.safe_load(handle)
with open(overlay_path) as handle:
    overlay = yaml.safe_load(handle) or {}
deep_update(cfg, overlay)
Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
with open(dest_path, "w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
}

mkdir -p "${ROOT_DIR}/logs/eval"
STAGING_DIR="${OUTPUT_DIR}/_ablation_configs"
mkdir -p "${STAGING_DIR}"

echo "Curia classifier ablations"
echo "  base:        ${BASE_CONFIG}"
echo "  recipes:     ${RUN_RECIPES[*]}"
echo "  data_dir:    ${DATA_DIR}"
echo "  annots:      ${ANNOTATIONS_DIR}"
echo "  output_dir:  ${OUTPUT_DIR}"
echo "  token cache: ${FEATURE_CACHE_DIR}"
echo "  weighted:    ${WEIGHTED_LOSS}"
echo "  n_folds:     ${N_FOLDS}"
echo "  seed:        ${SEED}"
echo

for recipe in "${RUN_RECIPES[@]}"; do
  overlay="${ABLATION_DIR}/${recipe}.yml"
  if [[ ! -f "${overlay}" ]]; then
    echo "ERROR: missing overlay ${overlay}" >&2
    echo "Known recipes: ${ALL_RECIPES[*]}" >&2
    exit 1
  fi

  recipe_config="${STAGING_DIR}/${recipe}.yml"
  recipe_output="${OUTPUT_DIR}/${recipe}"
  merge_config "${BASE_CONFIG}" "${overlay}" "${recipe_config}"

  echo "============================================================"
  echo "Curia ablation  recipe=${recipe}"
  echo "  config=${recipe_config}"
  echo "  output=${recipe_output}"
  echo "============================================================"

  python -m baselines.curia.classifier \
    --config "${recipe_config}" \
    --data-dir "${DATA_DIR}" \
    --annotations-dir "${ANNOTATIONS_DIR}" \
    --output-dir "${recipe_output}" \
    --feature-cache-dir "${FEATURE_CACHE_DIR}" \
    --n-folds "${N_FOLDS}" \
    --seed "${SEED}" \
    ${EXTRA_ARGS} \
    2>&1 | tee "${ROOT_DIR}/logs/eval/curia_ablation_${recipe}.log"

  echo "Finished ${recipe}"
  echo
done

echo "Curia classifier ablations complete!"
