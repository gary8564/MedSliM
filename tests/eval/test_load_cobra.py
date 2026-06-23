import yaml
import pytest
import torch
from accelerate import Accelerator

from med_slim.eval.load_cobra import load_cobra_from_experiment


def _testcase_cobra_config():
    return {
        "embed_dim": 32,
        "contrast_dim": 16,
        "input_dims": [64],
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "attn_dim": 16,
        "transformer_rotary_positional_encoding": None,
        "transformer_norm_first": True,
    }


def _testcase_experiment(
    tmp_path,
    *,
    regional_tokens=0,
    include_region_embed=False,
    embed_regional_tokens=None,
):
    exp_dir = tmp_path / "exp"
    ckpt_dir = exp_dir / "ckpt"
    ckpt_dir.mkdir(parents=True)

    config = {
        "cobra_config": {**_testcase_cobra_config(), "regional_tokens": regional_tokens},
        "sequence_encoder": "transformer",
        "pooling_target": "post_encoder",
    }
    (exp_dir / "config.yml").write_text(yaml.dump(config))

    state = {"cobra.attn.weight": torch.randn(1, 16, 32)}
    if include_region_embed:
        rt = embed_regional_tokens if embed_regional_tokens is not None else regional_tokens
        num_regions = 1 + rt
        state["cobra.region_embed.weight"] = torch.randn(num_regions, 32)
    torch.save(state, ckpt_dir / "classifier.pt")
    return exp_dir


def test_load_cobra_from_experiment(tmp_path):
    exp_dir = _testcase_experiment(
        tmp_path,
        regional_tokens=0,
        include_region_embed=True,
        embed_regional_tokens=4,
    )
    model, _ = load_cobra_from_experiment(str(exp_dir), Accelerator())
    assert model.regional_tokens == 4


def test_load_cobra_from_experiment_raises_on_regional_tokens_conflict(tmp_path):
    exp_dir = _testcase_experiment(tmp_path, regional_tokens=2, include_region_embed=True)
    ckpt_path = exp_dir / "ckpt" / "classifier.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state["cobra.region_embed.weight"] = torch.randn(5, 32)
    torch.save(state, ckpt_path)
    with pytest.raises(ValueError, match="conflicts with"):
        load_cobra_from_experiment(str(exp_dir), Accelerator())


def test_load_cobra_from_experiment_allows_tiled_without_region_embedding(tmp_path):
    exp_dir = _testcase_experiment(tmp_path, regional_tokens=4, include_region_embed=False)
    model, _ = load_cobra_from_experiment(str(exp_dir), Accelerator())
    assert model.regional_tokens == 4
    assert model.region_embed is None
