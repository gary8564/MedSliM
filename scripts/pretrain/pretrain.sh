#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --time=14:00:00                 
#SBATCH --job-name=pretrain
#SBATCH --output=logs/pretrain/stdout_pretrain_%j.txt    
#SBATCH --account=rwth1833     

### Setup
# Load Intel libraries (required by Triton for mamba_ssm kernels)
module load intel 2>/dev/null || true

source .venv/bin/activate

# Reduce fragmentation and enable expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Debug: Uncomment to see which parameters don't receive gradients in DDP
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Multi-GPU settings
NUM_GPUS=1

### Configuration
# Checkpoint to resume from (leave empty for training from scratch)
#RESUME_PATH="/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/test-run-MRNet/2026-02-08-18:35/medslim-epoch2000.pth.tar"

# Curriculum learning: load model weights only, reset optimizer and epoch.
# Set to true when adding new datasets or adding new slice encoder models.
#CURRICULUM=true

# Override slice encoder models from config
# Available: dinov2, dinov3, rad-dino, medsiglip, biomedclip, ark, mri-core
MODEL_NAMES="dinov2 dinov3 rad-dino medsiglip biomedclip ark"

# Stage feature caches to local SSD to avoid disk I/O during training for network latency.
# The training script caches all features in RAM after the first read.
# Staging to $TMPDIR speeds up that initial bulk read from ~50 min to ~2 min.
FEAT_BASE="/hpcwork/rwth1833/feat_caches"
STAGE_TO_LOCAL=true   # set to false to skip staging and read directly from /hpcwork
FEAT_CACHE_SUBDIR=("MRNet/slices_raw/crop" "KMAR-50K/slices_raw/adaptive" "fastMRI/slices_raw/adaptive")
if $STAGE_TO_LOCAL && [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ]; then
    LOCAL_BASE="$TMPDIR/feat_caches"
    echo "Staging feature caches to local SSD ($LOCAL_BASE)..."
    for ds_subdir in "${FEAT_CACHE_SUBDIR[@]}"; do
        for model in $MODEL_NAMES; do
            src="$FEAT_BASE/$ds_subdir/$model/train"
            dst="$LOCAL_BASE/$ds_subdir/$model/train"
            if [ -d "$src" ]; then
                mkdir -p "$dst"
                cp -a "$src/." "$dst/"
            fi
        done
    done
    echo "Data staging complete ($(du -sh "$LOCAL_BASE" | cut -f1))."
    export MEDSLIM_FEAT_BASE_OVERRIDE="$LOCAL_BASE"
fi

# Override view planes from config (space-separated, leave empty to use config defaults)
# Available: axial, sagittal, coronal
PLANES=""

# Sequence encoder and pooling
SEQUENCE_ENCODER="mamba2"   # mamba2 or transformer
POOLING="abmil"                    # abmil (default) or cls (requires transformer encoder)
USE_PACKED=true                   # true: packed sequences (no padding waste); false: random subsampling + padding

# Masked Slice Prediction (MSP) — JEPA-style auxiliary objective
# L = L_InfoNCE + lambda_mask * L_MSP + lambda_ctx * L_ctx
USE_MSP=true                      # true: enable MSP; false: pure InfoNCE (default)
MSP_LAMBDA_MASK=""                # Weight for MSP loss (leave empty for config default: 1.0)
MSP_LAMBDA_CTX=""                 # Weight for V-JEPA 2.1 context loss (leave empty for config default: 0.0)
MSP_MASK_RATIO=""                 # Contiguous masking ratio "min max" (leave empty for config default: "0.3 0.5")

### Build command arguments
EXTRA_ARGS=""
[[ -n "${RESUME_PATH}" ]]  && EXTRA_ARGS+=" --resume ${RESUME_PATH}"
[[ "${CURRICULUM}" == true ]] && EXTRA_ARGS+=" --curriculum"
[[ -n "${MODEL_NAMES}" ]]  && EXTRA_ARGS+=" --model-names ${MODEL_NAMES}"
[[ -n "${PLANES}" ]]       && EXTRA_ARGS+=" --planes ${PLANES}"
[[ "${USE_PACKED}" == true ]] && EXTRA_ARGS+=" --use-packed"
[[ "${USE_MSP}" == true ]]    && EXTRA_ARGS+=" --msp"
[[ -n "${MSP_LAMBDA_MASK}" ]] && EXTRA_ARGS+=" --msp-lambda-mask ${MSP_LAMBDA_MASK}"
[[ -n "${MSP_LAMBDA_CTX}" ]]  && EXTRA_ARGS+=" --msp-lambda-ctx ${MSP_LAMBDA_CTX}"
[[ -n "${MSP_MASK_RATIO}" ]]  && EXTRA_ARGS+=" --msp-mask-ratio ${MSP_MASK_RATIO}"

### Run
accelerate launch --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    ./med_slim/train/train.py \
    --sequence-encoder "${SEQUENCE_ENCODER}" \
    --pooling "${POOLING}" \
    ${EXTRA_ARGS}