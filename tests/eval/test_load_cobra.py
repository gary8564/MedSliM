import yaml
import pytest
import torch
from accelerate import Accelerator

from med_slim.eval.load_cobra import load_cobra_from_experiment, _resolve_per_fm_adapter


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
    include_within_slice_weights=False,
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
    if include_within_slice_weights:
        state["cobra.within_slice_agg.query.weight"] = torch.randn(1, 16)
    torch.save(state, ckpt_dir / "classifier.pt")
    return exp_dir


def test_load_cobra_from_experiment(tmp_path):
    exp_dir = _testcase_experiment(tmp_path)
    model, _ = load_cobra_from_experiment(str(exp_dir), Accelerator())
    assert model.pooling_target == "post_encoder"
    assert model.regional_tokens == 0


def test_load_cobra_from_experiment_preserves_flattened_regional_tokens(tmp_path):
    exp_dir = _testcase_experiment(tmp_path, regional_tokens=4)
    model, _ = load_cobra_from_experiment(str(exp_dir), Accelerator())
    assert model.regional_tokens == 4


def test_load_cobra_from_experiment_rejects_removed_within_slice_weights(tmp_path):
    exp_dir = _testcase_experiment(tmp_path, include_within_slice_weights=True)
    with pytest.raises(ValueError, match="removed within_slice_agg"):
        load_cobra_from_experiment(str(exp_dir), Accelerator())


def test_resolve_per_fm_adapter_defaults_to_per_dim():
    mode, dims = _resolve_per_fm_adapter({}, {"fm_router.fm_embed.weight": torch.randn(3, 8)})
    assert mode == "per_dim"
    assert dims is None


def test_resolve_per_fm_adapter_uses_metadata():
    meta = {"per_fm_adapter_mode": "per_fm_id", "fm_input_dims": [64, 128, 128]}
    weights = {
        "embed_fm.0.head.1.weight": torch.randn(32, 64),
        "embed_fm.1.head.1.weight": torch.randn(32, 128),
        "embed_fm.2.head.1.weight": torch.randn(32, 128),
    }
    mode, dims = _resolve_per_fm_adapter(meta, weights)
    assert mode == "per_fm_id"
    assert dims == [64, 128, 128]


def test_resolve_per_fm_adapter_infers_dims_from_weights():
    """Older checkpoints without metadata recover fm_input_dims from adapter weights."""
    weights = {
        "embed_fm.0.head.1.weight": torch.randn(32, 64),
        "embed_fm.2.head.1.weight": torch.randn(32, 128),
        "embed_fm.1.head.1.weight": torch.randn(32, 256),
        "embed_fm.0.head.0.weight": torch.randn(64),  # non-Linear key ignored
    }
    mode, dims = _resolve_per_fm_adapter({}, weights)
    assert mode == "per_fm_id"
    # Ordered by FM id: 0 -> 64, 1 -> 256, 2 -> 128.
    assert dims == [64, 256, 128]


def test_resolve_per_fm_adapter_supports_prefix():
    weights = {
        "cobra.embed_fm.0.head.1.weight": torch.randn(32, 64),
        "cobra.embed_fm.1.head.1.weight": torch.randn(32, 128),
    }
    mode, dims = _resolve_per_fm_adapter({}, weights, prefix="cobra.")
    assert mode == "per_fm_id"
    assert dims == [64, 128]
