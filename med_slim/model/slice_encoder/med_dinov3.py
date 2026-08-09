import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import hf_hub_download

from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


class MedDINOv3FeatureExtractor(nn.Module):
    """
    Slice-level CT feature extractor for MedDINOv3 ViT-B/16.

    The Hugging Face repo provides the checkpoint, while the architecture
    currently lives in the official GitHub repo. The simplest setup is to add
    that repo as a submodule at:

        med_slim/model/slice_encoder/MedDINOv3

    Initialize it with:

        git submodule update --init --recursive

    Input (shared with all MedSliM slice encoders):
        [B, C, W, H, D] grayscale volume (C=1), D = number of slices.

    Output:
        [B, D, 768] CLS-token embeddings, one embedding per slice.
    """

    DEFAULT_HF_REPO = "ricklisz123/MedDINOv3-ViTB-16-CT-3M"
    DEFAULT_FILENAME = "model.pth"
    DEFAULT_LOCAL_REPO = Path(__file__).resolve().parent / "MedDINOv3"
    DEFAULT_CHUNK_SIZE = 4

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        hf_repo: str = DEFAULT_HF_REPO,
        filename: str = DEFAULT_FILENAME,
        local_cache_dir: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """
        Args:
            checkpoint_path: Local MedDINOv3 checkpoint. If omitted, the
                checkpoint is downloaded from Hugging Face.
            hf_repo: Hugging Face repo containing the MedDINOv3 checkpoint.
            filename: Checkpoint filename inside the HF repo.
            local_cache_dir: Optional Hugging Face cache/download directory.
            chunk_size: Number of slices forwarded at once to reduce peak memory.
        """
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")

        self.hf_repo = hf_repo
        self.filename = filename
        self.chunk_size = chunk_size
        self.embed_dim = 768

        self._add_implementation_to_path()
        self.model = self._build_backbone()

        ckpt_path = self._resolve_checkpoint(
            checkpoint_path=checkpoint_path,
            local_cache_dir=local_cache_dir,
        )
        self._load_checkpoint(ckpt_path)
        logger.info(
            "MedDINOv3 loaded from %s: embed_dim=%d, chunk_size=%d, params=%s",
            ckpt_path,
            self.embed_dim,
            self.chunk_size,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

    @classmethod
    def _add_implementation_to_path(cls) -> None:
        repo_path = cls.DEFAULT_LOCAL_REPO.resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"MedDINOv3 submodule not found at {repo_path}. "
                "Run: git submodule update --init --recursive"
            )

        nnunet_path = repo_path / "nnUNet"
        dinov3_path = nnunet_path / "nnunetv2" / "training" / "nnUNetTrainer" / "dinov3"
        missing_paths = [path for path in (nnunet_path, dinov3_path) if not path.exists()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(
                f"MedDINOv3 submodule is missing expected package paths: {missing}. "
                "Run: git submodule update --init --recursive"
            )

        for path in (repo_path, nnunet_path, dinov3_path):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

    @staticmethod
    def _build_backbone() -> nn.Module:
        try:
            module = importlib.import_module(
                "nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer"
            )
            vit_base = module.vit_base
        except ImportError as exc:
            raise ImportError(
                "Could not import MedDINOv3's official ViT implementation from the bundled "
                "submodule. Ensure the submodule is initialized and its Python dependencies "
                f"are installed. Original error: {exc}"
            ) from exc

        return vit_base(
            drop_path_rate=0.0,
            layerscale_init=1.0e-5,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )

    def _resolve_checkpoint(
        self,
        checkpoint_path: Optional[str],
        local_cache_dir: Optional[str],
    ) -> Path:
        if checkpoint_path:
            path = Path(checkpoint_path).expanduser()
            if path.is_dir():
                path = path / self.filename
            if not path.exists():
                raise FileNotFoundError(f"MedDINOv3 checkpoint not found: {path}")
            return path

        cache_dir = None
        if local_cache_dir is not None:
            cache_dir = Path(local_cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)

        return Path(
            hf_hub_download(
                repo_id=self.hf_repo,
                filename=self.filename,
                cache_dir=cache_dir,
            )
        )

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("teacher", "state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported MedDINOv3 checkpoint format.")

        state_dict = {}
        for key, value in checkpoint.items():
            if "ibot" in key or "dino_head" in key:
                continue
            key = key.removeprefix("module.")
            key = key.removeprefix("backbone.")
            state_dict[key] = value
        return state_dict

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "MedDINOv3 checkpoint does not match the official ViT-B/16 backbone configuration."
            ) from exc

    @staticmethod
    def _extract_cls_features(outputs: Any) -> torch.Tensor:
        if not isinstance(outputs, dict) or "x_norm_clstoken" not in outputs:
            raise ValueError("MedDINOv3 forward_features must return a dict with 'x_norm_clstoken'.")

        features = outputs["x_norm_clstoken"]
        if not torch.is_tensor(features):
            raise ValueError(f"MedDINOv3 CLS token must be a tensor, got {type(features)!r}.")
        if features.ndim != 2:
            raise ValueError(f"MedDINOv3 CLS token must have shape [B, E], got {tuple(features.shape)}.")
        return features

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model.forward_features(x)
        return self._extract_cls_features(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Volume tensor [B, C, W, H, D], C=1.

        Returns:
            features: [B, D, 768].
        """
        x = x.swapaxes(2, -1)                         # [B, C, W, H, D] -> [B, C, D, H, W]
        B, C, *_ = x.shape
        assert C == 1, f"Expected grayscale input (C=1), got C={C}"

        x = rearrange(x, "b c d h w -> (b d) c h w")  # [(B*D), 1, H, W]
        x = x.repeat(1, 3, 1, 1)                      # MedDINOv3 ViT expects 3 channels

        outputs = []
        for start in range(0, x.shape[0], self.chunk_size):
            outputs.append(self._forward_chunk(x[start : start + self.chunk_size]))
        features = torch.cat(outputs, dim=0)
        return rearrange(features, "(b d) e -> b d e", b=B)


if __name__ == "__main__":
    model = MedDINOv3FeatureExtractor()
    model.to("cuda").eval()
    print(model)
    x = torch.randn(1, 1, 224, 224, 32).to("cuda")
    with torch.no_grad():
        y = model(x)
    print(y.shape)