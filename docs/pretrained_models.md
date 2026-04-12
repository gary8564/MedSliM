# Supported 2D Foundation Models

The following pretrained 2D foundation models are used as slice encoders to extract per-slice features:

| Model | Embedding Dim | Input Size | Source |
|-------|--------------|------------|--------|
| DINOv2-Large | 1024 | 224×224 | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) |
| DINOv3-ConvNeXt-Large | 1536 | 224×224 | [facebook/dinov3-convnext-large-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m) |
| RAD-DINO | 768 | 518×518 | [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) |
| MedSigLIP | 1152 | 448×448 | [google/medsiglip-448](https://huggingface.co/google/medsiglip-448) |
| BiomedCLIP | 512 | 224×224 | [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| Ark+ | 1376 | 768×768 | [Requires local checkpoint](https://github.com/Project-MONAI/research-contributions) |
| MRI-CORE | 768 | 224×224 | [Requires local checkpoint](https://github.com/Center-of-Medical-Imaging-and-AI/mri_foundation) |
| MedImageInsight | 1024 | 480×480 | [Requires local checkpoint](https://huggingface.co/microsoft/MedImageInsight) |

Models downloaded from HuggingFace are cached automatically. Models marked "Requires local checkpoint" need manual download; pass the checkpoint path via `--checkpoint` when running feature precomputation.
