import yaml
import pytest
import torch
from accelerate import Accelerator

from med_slim.eval.linear_classifier import SingleViewClassifier
from med_slim.eval.load_classifier import load_classifier_from_experiment
from med_slim.eval.load_cobra import _build_cobra


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


def _testcase_classifier_experiment(tmp_path, *, num_classes=3, include_head=True):
    """A checkpoint saved from a real SingleViewClassifier, so the keys are authentic."""
    exp_dir = tmp_path / "clf_exp"
    ckpt_dir = exp_dir / "fold_2" / "ckpt"
    ckpt_dir.mkdir(parents=True)

    cobra_cfg = _testcase_cobra_config()
    cobra = _build_cobra(cobra_cfg, "transformer", "abmil", "avg_pool", pooling_target="post_encoder")
    classifier = SingleViewClassifier(
        cobra_model=cobra,
        input_dim=cobra.output_dim,
        num_classes=num_classes,
        classifier_hidden_dim=8,
        classifier_dropout=0.1,
    )
    state = classifier.state_dict()
    if not include_head:
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    torch.save(state, ckpt_dir / "classifier.pt")

    config = {
        "cobra_config": cobra_cfg,
        "sequence_encoder": "transformer",
        "pooling_target": "post_encoder",
        "hyperparams": {"linear": {"hidden_dim": 8, "dropout": 0.1}},
    }
    (exp_dir / "config.yml").write_text(yaml.dump(config))
    return exp_dir, state


def test_load_classifier_from_experiment_restores_head_and_evals(tmp_path):
    exp_dir, saved = _testcase_classifier_experiment(tmp_path)

    model, cfg = load_classifier_from_experiment(str(exp_dir), Accelerator(), fold=2)

    assert model.num_classes == 3
    assert model.training is False
    assert cfg["pooling_target"] == "post_encoder"
    restored = model.state_dict()
    for key, tensor in saved.items():
        assert torch.equal(restored[key], tensor), key


def test_load_classifier_from_experiment_infers_binary_single_logit_head(tmp_path):
    exp_dir, _ = _testcase_classifier_experiment(tmp_path, num_classes=2)
    model, _ = load_classifier_from_experiment(str(exp_dir), Accelerator(), fold=2)
    assert model.num_classes == 2
    assert model.classifier.classifier[-1].out_features == 1


def test_load_classifier_from_experiment_rejects_headless_checkpoint(tmp_path):
    exp_dir, _ = _testcase_classifier_experiment(tmp_path, include_head=False)
    with pytest.raises(ValueError, match="No classifier.* keys found"):
        load_classifier_from_experiment(str(exp_dir), Accelerator(), fold=2)
