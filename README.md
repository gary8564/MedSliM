# MedSliM

**Med**ical **Sli**ce-wise **M**amba — Self-supervised pretraining for volumetric medical images from 2D foundation model features.

MedSliM extracts per-slice features from frozen 2D foundation models, then pretrains a COBRA encoder (Mamba2 sequence encoder + ABMIL pooling) via cross-foundation-model contrastive learning (MoCo). The resulting volume-level representations transfer to downstream classification via linear probing or k-NN evaluation.

## Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [Data Preparation](#data-preparation)
  - [1. Download and Preprocess Datasets](#1-download-and-preprocess-datasets)
  - [2. Precompute Slice Features](#2-precompute-slice-features)
- [Pretraining](#pretraining)
- [Evaluation](#evaluation)
  - [Linear Probing](#linear-probing)
  - [k-NN Classification](#k-nn-classification)
- [Visualization](#visualization)
  - [Slice Attention](#slice-attention)
  - [Embedding Clustering](#embedding-clustering)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

## Installation

### Prerequisites

- Python 3.12+
- CUDA 12.x with `nvcc`
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Optionally, install [mise](https://mise.jdx.dev/) for automatic environment activation:

```bash
curl https://mise.run | sh
```

### Setup

```bash
git clone https://github.com/gary8564/MedSliM.git && cd MedSliM
uv venv --python=3.12
source .venv/bin/activate

# Install PyTorch and build dependencies first
uv pip install torch==2.6.0 setuptools packaging wheel numpy==2.2.5 hatchling editables

# Install the package
uv sync --no-build-isolation
uv pip install -e .
```

If using mise:

```bash
mise trust   # auto-activates .venv and loads .env
```

### Troubleshooting

**`causal-conv1d` / `mamba-ssm` build failures:** These packages compile CUDA kernels and require `nvcc`. Ensure CUDA is available:

```bash
# On HPC clusters, you may need to load a CUDA module first:
# module load CUDA/12.4

export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

**`flash-attn` build failures:** Install a pre-built wheel instead:

```bash
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

Find the matching wheel for your setup at [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1).

## Overview

The MedSliM pipeline has three stages:

1. **Precompute slice features** — Run frozen 2D foundation models (DINOv2, RAD-DINO, MedSigLIP, BiomedCLIP, Ark+, MRI-CORE, etc.) on each slice of a 3D volume. Results are cached as `.safetensors` files.
2. **Self-supervised pretraining** — Train COBRA (Mamba2 sequence encoder + ABMIL pooling) using MoCo with cross-foundation-model contrastive pairs.
3. **Downstream evaluation** — Evaluate the frozen COBRA representations via linear probing or k-NN classification.

## Data Preparation

See [docs/data.md](docs/data.md) for dataset descriptions, download sources, and full preprocessing details.

### 1. Preprocess Datasets

```bash
# MRNet (pretraining + evaluation)
python scripts/preprocess_dataset/preprocess_MRNet.py \
    --data-dir /path/to/MRNet/MRNet-v1.0 \
    --save-dir /path/to/datasets/preprocessed/MRNet --workers 8

# fastMRI Knee (pretraining)
python scripts/preprocess_dataset/preprocess_fastMRI.py \
    --data-dir /path/to/fastMRI/knee \
    --save-dir /path/to/datasets/preprocessed/fastMRI --split train --workers 32

# KMAR-50K (pretraining)
python scripts/preprocess_dataset/preprocess_KMAR-50K.py \
    --data-dir /path/to/KMAR-50K \
    --save-dir /path/to/datasets/preprocessed/KMAR-50K --workers 8

# SKM-TEA (evaluation)
python scripts/preprocess_dataset/preprocess_SKM-TEA.py \
    --data-dir /path/to/SKM-TEA/qdess/v1-release \
    --save-dir /path/to/datasets/preprocessed/SKM-TEA --workers 8

# kneeMRI (evaluation)
python scripts/preprocess_dataset/preprocess_kneeMRI.py \
    --data-dir /path/to/kneeMRI \
    --save-dir /path/to/datasets/preprocessed/kneeMRI \
    --split train --plane sagittal --workers 8
python scripts/preprocess_dataset/preprocess_kneeMRI.py \
    --data-dir /path/to/kneeMRI \
    --save-dir /path/to/datasets/preprocessed/kneeMRI \
    --split test --plane sagittal --workers 8
```

### 2. Precompute Slice Features

Run frozen 2D foundation models on every slice. This only needs to run once per (dataset, model, plane) combination. Features are saved as `.safetensors` files:

```
/path/to/feat_caches/<DatasetName>/slices_raw/<spatial-mode>/<model>/<split>/<plane>/<uid>.safetensors
```

**HuggingFace models** (downloaded automatically):

```bash
python scripts/precompute_slice_feature.py \
    --data-dir /path/to/datasets/preprocessed/MRNet \
    --save-dir /path/to/feat_caches/MRNet \
    --model-name dinov2 \
    --plane sagittal --split train \
    --spatial-mode crop --use-raw-slice-resolution --amp bf16
```

**Models requiring a local checkpoint** (`ark`, `mri-core`, `medimageinsight`):

```bash
python scripts/precompute_slice_feature.py \
    --data-dir /path/to/datasets/preprocessed/MRNet \
    --save-dir /path/to/feat_caches/MRNet \
    --model-name ark --checkpoint /path/to/models/ark/Ark6_swinLarge768_ep50.pth.tar \
    --plane sagittal --split train \
    --spatial-mode crop --use-raw-slice-resolution --amp bf16 --workers 2
```

**Parallel sharding for large datasets:**

```bash
for SHARD_ID in 0 1 2 3; do
    python scripts/precompute_slice_feature.py \
        --data-dir /path/to/datasets/preprocessed/fastMRI \
        --save-dir /path/to/feat_caches/fastMRI \
        --model-name medsiglip \
        --plane sagittal --split train \
        --spatial-mode adaptive --use-raw-slice-resolution --amp bf16 \
        --shard-id $SHARD_ID --num-shards 4 --workers 1 &
done
wait
```

**Available arguments:**

| Argument | Description |
|----------|-------------|
| `--data-dir` | Preprocessed dataset root (required) |
| `--save-dir` | Output directory for feature caches (required) |
| `--model-name` | Slice encoder: `dinov2`, `dinov3`, `rad-dino`, `medsiglip`, `biomedclip`, `ark`, `mri-core`, `medimageinsight` |
| `--plane` | Anatomical plane: `sagittal`, `coronal`, `axial` |
| `--split` | Dataset split: `train`, `val`, `test` |
| `--spatial-mode` | Preprocessing: `resize`, `resample`, `crop`, `adaptive` |
| `--use-raw-slice-resolution` | Keep original slice count (variable-length sequences) |
| `--num-slices` | Fixed number of slices (ignored if `--use-raw-slice-resolution`) |
| `--amp` | Mixed precision: `fp16` or `bf16` (recommended) |
| `--checkpoint` | Local checkpoint path (required for `ark`, `mri-core`, `medimageinsight`) |
| `--workers` | DataLoader workers |
| `--min-slices` | Skip volumes with fewer slices than this |
| `--shard-id` / `--num-shards` | Parallel sharding across multiple processes |

## Pretraining

Train COBRA via MoCo cross-foundation-model contrastive learning. Requires precomputed features from **at least 2** foundation models.

Update feature cache paths in `med_slim/configs/pretrain.yml`, then run:

```bash
# Single GPU
accelerate launch --num_processes=1 --mixed_precision=bf16 \
    med_slim/train/train.py \
    --sequence-encoder mamba2 --pooling abmil --use-packed \
    --model-names dinov2 dinov3 rad-dino medsiglip biomedclip ark

# Multi-GPU (DDP)
accelerate launch --num_processes=4 --mixed_precision=bf16 \
    med_slim/train/train.py \
    --sequence-encoder mamba2 --pooling abmil --use-packed \
    --model-names dinov2 dinov3 rad-dino medsiglip biomedclip ark

# Resume from checkpoint
accelerate launch --num_processes=1 --mixed_precision=bf16 \
    med_slim/train/train.py \
    --sequence-encoder mamba2 --pooling abmil --use-packed \
    --resume /path/to/checkpoints/medslim-epoch2000.pth.tar

# Curriculum learning (load weights, reset optimizer/epoch for new datasets)
accelerate launch --num_processes=1 --mixed_precision=bf16 \
    med_slim/train/train.py \
    --sequence-encoder mamba2 --pooling abmil --use-packed \
    --resume /path/to/checkpoints/medslim-epoch2000.pth.tar --curriculum
```

**Available arguments:**

| Argument | Description |
|----------|-------------|
| `--sequence-encoder` | `mamba2` or `transformer` |
| `--pooling` | `abmil` or `cls` (`cls` requires `transformer`) |
| `--use-packed` | Packed variable-length sequences (no padding waste) |
| `--model-names` | Override FM models from config (space-separated) |
| `--planes` | Override planes from config (space-separated) |
| `--resume` | Path to checkpoint to resume training |
| `--curriculum` | Load weights from `--resume` but reset optimizer/epoch |
| `-c` / `--config` | Config file path |

>[!NOTE] Staging features to local SSD (recommended on HPC clusters):
>Precomputed `.safetensors` feature files are read repeatedly across epochs. On HPC clusters where the parallel filesystem has high latency under concurrent load, staging these files to a local NVMe SSD before training significantly reduces I/O wait and compute waste.

>[!NOTE] Checkpoints are saved every 50 epochs to the path in `pretrain.yml`. Training is logged to [Weights & Biases](https://wandb.ai/).

## Inference

### Linear Probing

Train a linear classifier on frozen COBRA embeddings with k-fold cross-validation.

Update `med_slim/configs/linear_classifier.yml` with your paths, then run:

```bash
python med_slim/eval/linear_classifier.py \
    --linear-classifier-config med_slim/configs/linear_classifier.yml \
    --checkpoint-path /path/to/checkpoints/medslim-epoch2000.pth.tar \
    --fm-model-names "mri-core medimageinsight ark dinov2 dinov3 rad-dino biomedclip medsiglip" \
    --sequence-encoder mamba2 \
    --fm-pooling avg_pool \
    --pooling-target raw \
    --weighted-loss \
    --n-folds 3
```

**Available arguments:**

| Argument | Description |
|----------|-------------|
| `--linear-classifier-config` | Config file path (required) |
| `--checkpoint-path` | COBRA checkpoint (overrides config) |
| `--fm-model-names` | FM models to use (space-separated string) |
| `--sequence-encoder` | `mamba2` or `transformer` |
| `--fm-pooling` | `avg_pool` or `attention` |
| `--pooling-target` | Representation level: `raw`, `post_embed`, `post_encoder` |
| `--slice-pooling` | `abmil` (for mamba2) or `cls` (for transformer) |
| `--weighted-loss` | Use class-weighted loss for imbalanced datasets |
| `--fine-tune` | Fine-tune COBRA backbone (not just linear head) |
| `--n-folds` | Number of cross-validation folds |

### k-NN Classification

```bash
python med_slim/eval/knn_classifier.py \
    --config med_slim/configs/linear_classifier.yml \
    --checkpoint-path /path/to/checkpoints/medslim-epoch2000.pth.tar \
    --fm-model-names "mri-core medimageinsight ark" \
    --sequence-encoder mamba2 \
    --pooling-target raw \
    --nb-knn 10 20 50 100 200 \
    --temperature 0.07
```

## Visualization

### Slice Attention

Visualize per-slice ABMIL attention weights:

```bash
# From a fine-tuned experiment directory
python -m med_slim.eval.slice_attention \
    --experiment-dir /path/to/experiments/<experiment_folder> \
    --fold 3 --dataset-name SKM-TEA --split test --batch-size 8

# From a pretrained checkpoint (no fine-tuning)
python -m med_slim.eval.slice_attention \
    --checkpoint-path /path/to/checkpoints/medslim-epoch2000.pth.tar \
    --feat-dir /path/to/feat_caches/MRNet/slices_raw/crop \
    --annotations-path /path/to/datasets/preprocessed/MRNet/test.csv \
    --output-dir /path/to/experiments/slice_attention_MRNet \
    --dataset-name MRNet --plane sagittal --fm-model-names "mri-core" --split test
```

### Embedding Clustering

Visualize COBRA embeddings with UMAP/t-SNE:

```bash
# Cross-dataset (color by plane)
python -m med_slim.eval.embed_cluster \
    --multi-dataset \
    --checkpoint-path /path/to/checkpoints/medslim-epoch2000.pth.tar \
    --fm-model-names "mri-core medimageinsight ark" \
    --output-dir /path/to/experiments/embed_cluster \
    --method umap --pooling-target post_embed --save-embeddings

# Single dataset (color by pathology)
python -m med_slim.eval.embed_cluster \
    --checkpoint-path /path/to/checkpoints/medslim-epoch2000.pth.tar \
    --feat-dir /path/to/feat_caches/MRNet/slices_raw/crop \
    --annotations-dir /path/to/datasets/preprocessed/MRNet \
    --output-dir /path/to/experiments/embed_cluster \
    --dataset-name MRNet --plane sagittal \
    --fm-model-names "mri-core medimageinsight ark" \
    --target-labels abnormal --task binary \
    --method umap --pooling-target raw --save-embeddings
```

## Configuration

YAML configuration files live in `med_slim/configs/`. Update placeholders (`/path/to/...`) before running.

| File | Purpose |
|------|---------|
| `pretrain.yml` | SSL pretraining: datasets, foundation models, COBRA architecture, training hyperparameters |
| `linear_classifier.yml` | Linear probing / fine-tuning: checkpoint, dataset, training setup |
| `eval_datasets.yaml` | Downstream dataset metadata: task types, label columns, display names |

CLI arguments override config values when both are provided.

## Project Structure

```
MedSliM/
├── med_slim/
│   ├── configs/                        # YAML configuration files
│   ├── data/                           # Dataset and dataloader classes
│   │   ├── slice_dataset.py            # NIfTI volume loading
│   │   └── feat_dataset.py             # Precomputed feature loading
│   ├── eval/                           # Evaluation scripts
│   │   ├── linear_classifier.py        # Linear probing with k-fold CV
│   │   ├── knn_classifier.py           # k-NN classification
│   │   ├── slice_attention.py          # Attention visualization
│   │   ├── embed_cluster.py            # UMAP/t-SNE embedding visualization
│   │   └── load_cobra.py               # Checkpoint loading utility
│   ├── model/
│   │   ├── sequence_encoder/cobra.py   # COBRA: Mamba2/Transformer + ABMIL
│   │   ├── ssl/moco.py                 # MoCo v3 self-supervised wrapper
│   │   └── slice_encoder/              # 2D foundation model wrappers
│   ├── train/train.py                  # Pretraining entry point
│   └── utils/
│       └── preprocessing/              # Transforms, augmentation
├── scripts/
│   ├── precompute_slice_feature.py     # Feature extraction CLI
│   └── preprocess_dataset/             # Dataset-specific NIfTI conversion
├── tests/                              # Unit and integration tests
├── docs/data.md                        # Dataset documentation
└── pyproject.toml
```
