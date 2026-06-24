#!/usr/bin/bash
#
# Patch slice_spacing_mm metadata into all precomputed feature caches.
# CPU-only, no GPU needed. Runs in ~5-15 min depending on I/O.

### SLURM defaults (ignored if run interactively)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=c23mm
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:30:00
#SBATCH --job-name=patch_spacing
#SBATCH --output=logs/patch_spacing_%j.txt

source .venv/bin/activate

SCRIPT="scripts/preprocess_dataset/patch_slice_spacing_metadata.py"
WORKERS=16

echo "=== Patching MRNet ==="
python "$SCRIPT" \
    --feat-dir "/hpcwork/rwth1833/feat_caches/MRNet/slices_raw/crop" \
    --data-dir "/hpcwork/rwth1833/datasets/preprocessed/MRNet" \
    --split train --workers "$WORKERS"

echo ""
echo "=== Patching fastMRI ==="
python "$SCRIPT" \
    --feat-dir "/hpcwork/rwth1833/feat_caches/fastMRI/slices_raw/adaptive" \
    --data-dir "/hpcwork/rwth1833/datasets/preprocessed/fastMRI" \
    --split train --workers "$WORKERS"

echo ""
echo "=== Patching KMAR-50K ==="
python "$SCRIPT" \
    --feat-dir "/hpcwork/rwth1833/feat_caches/KMAR-50K/slices_raw/adaptive" \
    --data-dir "/hpcwork/rwth1833/datasets/preprocessed/KMAR-50K" \
    --split train --workers "$WORKERS"

echo ""
echo "=== Verification: spot-check metadata ==="
python -c "
from safetensors import safe_open
import glob, os

for name, base in [
    ('MRNet', '/hpcwork/rwth1833/feat_caches/MRNet/slices_raw/crop'),
    ('fastMRI', '/hpcwork/rwth1833/feat_caches/fastMRI/slices_raw/adaptive'),
    ('KMAR-50K', '/hpcwork/rwth1833/feat_caches/KMAR-50K/slices_raw/adaptive'),
]:
    files = sorted(glob.glob(os.path.join(base, 'dinov2/train/sagittal/*.safetensors')))[:3]
    for f in files:
        with safe_open(f, framework='pt', device='cpu') as sf:
            m = sf.metadata()
        spacing = m.get('slice_spacing_mm', 'MISSING')
        print(f'{name}: uid={m[\"uid\"]} plane={m[\"plane\"]} spacing={spacing}mm')
"

echo ""
echo "=== All done ==="
