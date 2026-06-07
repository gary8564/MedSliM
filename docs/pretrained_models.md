# Pretrained 2D Slice Extractor

## Supported 2D Foundation Models

The following pretrained 2D foundation models are used as slice encoders to extract per-slice features:

| Model                 | Embedding Dim | Input Size | Source                                                                                                                                      |
| --------------------- | ------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| DINOv2-Large          | 1024          | 224×224    | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large)                                                                       |
| DINOv3-ConvNeXt-Large | 1536          | 224×224    | [facebook/dinov3-convnext-large-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m)                 |
| RAD-DINO              | 768           | 518×518    | [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino)                                                                             |
| MedSigLIP             | 1152          | 448×448    | [google/medsiglip-448](https://huggingface.co/google/medsiglip-448)                                                                         |
| BiomedCLIP            | 512           | 224×224    | [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| Ark+                  | 1376          | 768×768    | [Requires local checkpoint](https://github.com/jlianglab/Ark)                                                                               |
| MRI-CORE              | 768           | 224×224    | [Requires local checkpoint](https://github.com/mazurowski-lab/mri_foundation)                                                               |
| MedImageInsight       | 1024          | 480×480    | [Requires local checkpoint](https://huggingface.co/lion-ai/MedImageInsights)                                                                |

## Precompute Slice Features

Models downloaded from HuggingFace are cached automatically. Models marked "Requires local checkpoint" need manual download; pass the checkpoint path via `--checkpoint` when running feature precomputation.

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

| Argument                      | Description                                                                                                    |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `--data-dir`                  | Preprocessed dataset root (required)                                                                           |
| `--save-dir`                  | Output directory for feature caches (required)                                                                 |
| `--model-name`                | Slice encoder: `dinov2`, `dinov3`, `rad-dino`, `medsiglip`, `biomedclip`, `ark`, `mri-core`, `medimageinsight` |
| `--plane`                     | Anatomical plane: `sagittal`, `coronal`, `axial`                                                               |
| `--split`                     | Dataset split: `train`, `val`, `test`                                                                          |
| `--spatial-mode`              | Preprocessing: `resize`, `resample`, `crop`, `adaptive`                                                        |
| `--use-raw-slice-resolution`  | Keep original slice count (variable-length sequences)                                                          |
| `--num-slices`                | Fixed number of slices (ignored if `--use-raw-slice-resolution`)                                               |
| `--amp`                       | Mixed precision: `fp16` or `bf16` (recommended)                                                                |
| `--checkpoint`                | Local checkpoint path (required for `ark`, `mri-core`, `medimageinsight`)                                      |
| `--workers`                   | DataLoader workers                                                                                             |
| `--min-slices`                | Skip volumes with fewer slices than this                                                                       |
| `--shard-id` / `--num-shards` | Parallel sharding across multiple processes                                                                    |
