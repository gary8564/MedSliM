import yaml
import pytest
import torch
from accelerate import Accelerator

from med_slim.eval.load_cobra import (
    load_cobra_from_experiment,
    _resolve_per_fm_adapter,
    resolve_raw_aggregation_fm,
)


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


def test_load_cobra_from_experiment_restores_raw_aggregation_index(tmp_path):
    """A saved 'raw' experiment config restores its explicit raw_aggregation_index/dim."""
    exp_dir = tmp_path / "exp_raw"
    ckpt_dir = exp_dir / "ckpt"
    ckpt_dir.mkdir(parents=True)

    config = {
        "cobra_config": {**_testcase_cobra_config()},
        "sequence_encoder": "transformer",
        "pooling_target": "raw",
        "raw_output_dim": 96,
        "raw_aggregation_index": 1,
        "raw_aggregation_fm": "fm-b",
    }
    (exp_dir / "config.yml").write_text(yaml.dump(config))
    state = {"cobra.attn.weight": torch.randn(1, 16, 32)}
    torch.save(state, ckpt_dir / "classifier.pt")

    model, cfg = load_cobra_from_experiment(str(exp_dir), Accelerator())
    assert model.pooling_target == "raw"
    assert model.raw_aggregation_index == 1
    assert model._raw_output_dim == 96
    assert cfg["raw_aggregation_fm"] == "fm-b"


# resolve_raw_aggregation_fm
def _pretrain_cfg_with_fms(fm_dims: dict) -> dict:
    return {
        "model": {
            "slice_encoder_models": [
                {"name": name, "embed_dim": dim} for name, dim in fm_dims.items()
            ]
        }
    }


def test_resolve_raw_aggregation_fm_returns_none_when_not_raw():
    cfg = _pretrain_cfg_with_fms({"mri-core": 768})
    result = resolve_raw_aggregation_fm("post_embed", ["mri-core"], cfg, None)
    assert result == (None, None, None)


def test_resolve_raw_aggregation_fm_single_fm_infers_automatically():
    cfg = _pretrain_cfg_with_fms({"mri-core": 768})
    index, dim, name = resolve_raw_aggregation_fm("raw", ["mri-core"], cfg, None)
    assert (index, dim, name) == (0, 768, "mri-core")


def test_resolve_raw_aggregation_fm_single_fm_mismatch_raises():
    cfg = _pretrain_cfg_with_fms({"mri-core": 768})
    with pytest.raises(ValueError, match="does not match the only evaluated FM"):
        resolve_raw_aggregation_fm("raw", ["mri-core"], cfg, "curia")


def test_resolve_raw_aggregation_fm_multi_fm_requires_explicit_name():
    cfg = _pretrain_cfg_with_fms({"mri-core": 768, "curia": 1024})
    with pytest.raises(ValueError, match="requires an explicit raw_aggregation_fm"):
        resolve_raw_aggregation_fm("raw", ["mri-core", "curia"], cfg, None)


def test_resolve_raw_aggregation_fm_multi_fm_unknown_name_raises():
    cfg = _pretrain_cfg_with_fms({"mri-core": 768, "curia": 1024})
    with pytest.raises(ValueError, match="must be one of the evaluated FMs"):
        resolve_raw_aggregation_fm("raw", ["mri-core", "curia"], cfg, "dinov2")


def test_resolve_raw_aggregation_fm_multi_fm_resolves_index_and_dim():
    cfg = _pretrain_cfg_with_fms({"mri-core": 768, "curia": 1024})
    index, dim, name = resolve_raw_aggregation_fm(
        "raw", ["mri-core", "curia"], cfg, "curia"
    )
    assert (index, dim, name) == (1, 1024, "curia")


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
