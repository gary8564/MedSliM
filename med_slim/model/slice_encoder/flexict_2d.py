import importlib
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


class FlexiCT2DFeatureExtractor(nn.Module):
    """
    Slice-level CT feature extractor for FlexiCT-2D.

    The official FlexiCT repository provides the backbone code. Keep it checked
    out next to this file at:

        med_slim/model/slice_encoder/FlexiCT

    Input (shared with all MedSliM slice encoders):
        [B, C, W, H, D] grayscale volume (C=1), D = number of slices.

    Output:
        [B, D, 864] CLS-token embeddings, one embedding per slice.

    Memory note:
        FlexiCT-2D's backbone uses patch_size=8, so it produces 4x as many patch tokens per slice at the
        same resolution. 
        All D slices of a volume are forwarded through the transformer as a single batch, 
        so peak GPU memory scales with the number of slices in the deepest volume in the dataset, 
        not just the in-plane resolution. 
        `chunk_size` bounds this by forwarding at most `chunk_size` slices through the backbone at once.
    """

    DEFAULT_CHECKPOINT = Path("/hpcwork/rwth1833/models/FlexiCT-2D.pth")
    DEFAULT_LOCAL_REPO = Path(__file__).resolve().parent / "FlexiCT"
    DEFAULT_CHUNK_SIZE = 4

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """
        Args:
            checkpoint_path: Local FlexiCT-2D checkpoint. Defaults to
                /hpcwork/rwth1833/models/FlexiCT-2D.pth.
            chunk_size: Number of slices forwarded through the backbone at
                once, to bound peak GPU memory on deep volumes.
        """
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")

        self.chunk_size = chunk_size
        self._add_implementation_to_path()
        ckpt_path = self._resolve_checkpoint(checkpoint_path)
        self.model = self._build_model(ckpt_path)
        logger.info(
            "FlexiCT-2D loaded from %s: chunk_size=%d, params=%s",
            ckpt_path,
            self.chunk_size,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

    @classmethod
    def _add_implementation_to_path(cls) -> None:
        repo_path = cls.DEFAULT_LOCAL_REPO.resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"FlexiCT repository not found at {repo_path}. "
                "Clone with `git submodule update --init --recursive` to add it as a submodule."
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

    @staticmethod
    def _build_model(checkpoint_path: Path) -> nn.Module:
        try:
            module = importlib.import_module("flexi_ct")
            flexi_ct_2d = module.Flexi_CT_2D
        except ImportError as exc:
            raise ImportError(
                "Could not import FlexiCT's official Flexi_CT_2D implementation from "
                "med_slim/model/slice_encoder/FlexiCT. Original error: "
                f"{exc}"
            ) from exc

        return flexi_ct_2d(checkpoint_path=str(checkpoint_path), device="cpu")

    @classmethod
    def _resolve_checkpoint(cls, checkpoint_path: Optional[str]) -> Path:
        path = Path(checkpoint_path).expanduser() if checkpoint_path else cls.DEFAULT_CHECKPOINT
        if not path.exists():
            raise FileNotFoundError(
                f"FlexiCT-2D checkpoint not found: {path}. "
                "Pass checkpoint=... to build_slice_encoder or place it at the default path."
            )
        return path

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        features = outputs["cls_token"]
        if not torch.is_tensor(features):
            raise ValueError(f"FlexiCT-2D CLS token must be a tensor, got {type(features)!r}.")
        if features.ndim != 2:
            raise ValueError(f"FlexiCT-2D CLS token must have shape [B, E], got {tuple(features.shape)}.")
        return features

    def _forward_slices(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, x.shape[0], self.chunk_size):
            outputs.append(self._forward_chunk(x[start : start + self.chunk_size]))
        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Volume tensor [B, C, W, H, D], C=1.

        Returns:
            features: [B, D, 864].
        """
        x = x.swapaxes(2, -1)                         # [B, C, W, H, D] -> [B, C, D, H, W]
        B, C, _, H, W = x.shape
        assert C == 1, f"Expected grayscale input (C=1), got C={C}"
        if H % 8 != 0 or W % 8 != 0:
            raise ValueError(f"FlexiCT-2D expects H and W divisible by 8, got {(H, W)}.")

        x = rearrange(x, "b c d h w -> (b d) c h w")  # [(B*D), 1, H, W]

        features = self._forward_slices(x)
        return rearrange(features, "(b d) e -> b d e", b=B)

if __name__ == "__main__":
    model = FlexiCT2DFeatureExtractor()
    model.to("cuda").eval()
    print(model)
    x = torch.randn(1, 1, 224, 224, 32).to("cuda")
    with torch.no_grad():
        y = model(x)
    print(y.shape)