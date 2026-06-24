#!/usr/bin/bash

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --partition=c23g
#SBATCH --time=24:00:00                 
#SBATCH --job-name=pretrain
#SBATCH --output=logs/pretrain/stdout_pretrain_router_%j.txt    
#SBATCH --account=p0021834     

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
# RESUME_PATH="/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-03-24-16:00/medslim-epoch1750.pth.tar"

# Curriculum learning: load model weights only, reset optimizer and epoch.
# Set to true when adding new datasets or adding new slice encoder models.
#CURRICULUM=true

# Override slice encoder models from config
# Available: curia dinov2, dinov3, rad-dino, medsiglip, biomedclip, ark, mri-core, medimageinsight
# Keep this order fixed for router runs; it is saved as fm_id_order in checkpoints.
MODEL_NAMES="dinov2 dinov3 rad-dino medsiglip biomedclip ark mri-core medimageinsight curia"

# Stage feature caches to local SSD to avoid slow initial reads from shared storage.
# The training script caches all features in RAM after the first read.
# Staging happens inside the GPU allocation, so it is billed. Keep it only when
# copy + RAM preload is faster than direct /hpcwork preload (true for full 3-dataset runs).
FEAT_BASE="/hpcwork/rwth1833/feat_caches"
STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-true}"   # set to false to skip staging and read directly from /hpcwork
STAGE_PARALLEL_JOBS="${STAGE_PARALLEL_JOBS:-6}"  # parallel copy jobs; avoid overloading shared FS
FEAT_CACHE_SUBDIR=("MRNet/slices_raw/crop" "KMAR-50K/slices_raw/adaptive" "fastMRI/slices_raw/adaptive")

# Override view planes from config (space-separated, leave empty to use config defaults)
# Available: axial, sagittal, coronal
PLANES=""

if $STAGE_TO_LOCAL && [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ]; then
    LOCAL_BASE="$TMPDIR/feat_caches"
    stage_start=$(date +%s)
    echo "Staging feature caches to local SSD ($LOCAL_BASE) with ${STAGE_PARALLEL_JOBS} parallel copy jobs..."

    stage_one() {
        local src="$1"
        local dst="$2"
        if [ -d "$src" ]; then
            mkdir -p "$dst"
            cp -a "$src/." "$dst/"
        fi
    }

    active_jobs=0
    for ds_subdir in "${FEAT_CACHE_SUBDIR[@]}"; do
        for model in $MODEL_NAMES; do
            src_train="$FEAT_BASE/$ds_subdir/$model/train"
            dst_train="$LOCAL_BASE/$ds_subdir/$model/train"
            if [ -n "$PLANES" ]; then
                for plane in $PLANES; do
                    stage_one "$src_train/$plane" "$dst_train/$plane" &
                    active_jobs=$((active_jobs + 1))
                    if [ "$active_jobs" -ge "$STAGE_PARALLEL_JOBS" ]; then
                        wait -n
                        active_jobs=$((active_jobs - 1))
                    fi
                done
            else
                stage_one "$src_train" "$dst_train" &
                active_jobs=$((active_jobs + 1))
                if [ "$active_jobs" -ge "$STAGE_PARALLEL_JOBS" ]; then
                    wait -n
                    active_jobs=$((active_jobs - 1))
                fi
            fi
        done
    done
    wait

    stage_elapsed=$(( $(date +%s) - stage_start ))
    echo "Data staging complete ($(du -sh "$LOCAL_BASE" | cut -f1), ${stage_elapsed}s)."
    export MEDSLIM_FEAT_BASE_OVERRIDE="$LOCAL_BASE"
else
    echo "STAGE_TO_LOCAL=false or TMPDIR unavailable; reading feature caches from $FEAT_BASE."
fi

# Sequence encoder and pooling
SEQUENCE_ENCODER="mamba2"   # mamba2 or transformer
POOLING="abmil"    # abmil (default) or cls (requires transformer encoder)
USE_PACKED=false                   # true: packed sequences (no padding waste); false: evenly-spaced subsampling + padding
PHYSICAL_PE="${PHYSICAL_PE:-false}"  # true: sinusoidal PE from normalized relative slice depth [0, 1]
NUM_EPOCHS="${NUM_EPOCHS:-}"         # leave empty for config default (e.g. 2000)
REGIONAL_TOKENS="${REGIONAL_TOKENS:-0}"  # 0 = global CLS; 4 = global + 2x2 regional CLS flattened into sequence

# FM fusion: how to combine FM embeddings per slice.
FM_POOLING="router"               # avg_pool | router
PER_FM_ADAPTER_MODE="per_dim"   # per_dim | per_fm_id (leave empty to use config default)
SSL_FM_MODE="subset"              # pair (cross-FM baseline) | subset (FM-set router/avg_pool SSL)
FM_SUBSET_SIZE=4                  # FMs per view in subset mode (must be < num FMs)
FM_SUBSET_MIN_OVERLAP=0           # min shared FMs between the two views
FM_SUBSET_MAX_OVERLAP=0           # max shared FMs between the two views
ROUTER_MODE="soft"                # soft | topk (topk is a later ablation)
ROUTER_TOP_K=""                   # k for topk routing (leave empty for soft)
ROUTER_TEMPERATURE="0.7"          # keep soft routing; lower values were less transferable
ROUTER_LOAD_BALANCE_WEIGHT=""     # leave empty for config default
ROUTER_Z_LOSS_WEIGHT=""           # leave empty for config default (0.0)
ROUTER_CONFIDENCE_WEIGHT=""       # leave empty for config default (0.0)

### Build command arguments
EXTRA_ARGS=""
[[ -n "${RESUME_PATH}" ]]  && EXTRA_ARGS+=" --resume ${RESUME_PATH}"
[[ "${CURRICULUM}" == true ]] && EXTRA_ARGS+=" --curriculum"
[[ -n "${MODEL_NAMES}" ]]  && EXTRA_ARGS+=" --model-names ${MODEL_NAMES}"
[[ -n "${PLANES}" ]]       && EXTRA_ARGS+=" --planes ${PLANES}"
[[ "${USE_PACKED}" == true ]] && EXTRA_ARGS+=" --use-packed"
[[ "${PHYSICAL_PE}" == true ]] && EXTRA_ARGS+=" --physical-pe"
[[ -n "${NUM_EPOCHS}" ]] && EXTRA_ARGS+=" --num-epochs ${NUM_EPOCHS}"
[[ "${REGIONAL_TOKENS}" -gt 0 ]] && EXTRA_ARGS+=" --regional-tokens ${REGIONAL_TOKENS}"
[[ -n "${FM_POOLING}" ]]      && EXTRA_ARGS+=" --fm-pooling ${FM_POOLING}"
[[ -n "${PER_FM_ADAPTER_MODE}" ]] && EXTRA_ARGS+=" --per-fm-adapter-mode ${PER_FM_ADAPTER_MODE}"
[[ -n "${SSL_FM_MODE}" ]]     && EXTRA_ARGS+=" --ssl-fm-mode ${SSL_FM_MODE}"
[[ -n "${FM_SUBSET_SIZE}" ]]  && EXTRA_ARGS+=" --fm-subset-size ${FM_SUBSET_SIZE}"
[[ -n "${FM_SUBSET_MIN_OVERLAP}" ]] && EXTRA_ARGS+=" --fm-subset-min-overlap ${FM_SUBSET_MIN_OVERLAP}"
[[ -n "${FM_SUBSET_MAX_OVERLAP}" ]] && EXTRA_ARGS+=" --fm-subset-max-overlap ${FM_SUBSET_MAX_OVERLAP}"
[[ -n "${ROUTER_MODE}" ]]     && EXTRA_ARGS+=" --router-mode ${ROUTER_MODE}"
[[ -n "${ROUTER_TOP_K}" ]]    && EXTRA_ARGS+=" --router-top-k ${ROUTER_TOP_K}"
[[ -n "${ROUTER_TEMPERATURE}" ]] && EXTRA_ARGS+=" --router-temperature ${ROUTER_TEMPERATURE}"
[[ -n "${ROUTER_LOAD_BALANCE_WEIGHT}" ]] && EXTRA_ARGS+=" --router-load-balance-weight ${ROUTER_LOAD_BALANCE_WEIGHT}"
[[ -n "${ROUTER_Z_LOSS_WEIGHT}" ]] && EXTRA_ARGS+=" --router-z-loss-weight ${ROUTER_Z_LOSS_WEIGHT}"
[[ -n "${ROUTER_CONFIDENCE_WEIGHT}" ]] && EXTRA_ARGS+=" --router-confidence-weight ${ROUTER_CONFIDENCE_WEIGHT}"

### Run
accelerate launch --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    ./med_slim/train/train.py \
    --sequence-encoder "${SEQUENCE_ENCODER}" \
    --pooling "${POOLING}" \
    ${EXTRA_ARGS}