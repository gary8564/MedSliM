import argparse
import os

import pytest
import yaml

from med_slim.train.train import adjust_msp_ctx_lambda


def _parse_msp_args(argv: list[str]) -> argparse.Namespace:
    """Build just the MSP-related argparse arguments and parse."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--msp", action="store_true")
    parser.add_argument("--msp-lambda-mask", type=float, default=None)
    parser.add_argument("--msp-lambda-ctx", type=float, default=None)
    parser.add_argument(
        "--msp-mask-ratio", nargs=2, type=float, default=None,
        metavar=("MIN", "MAX"),
    )
    return parser.parse_args(argv)

def _apply_msp_overrides(args, cfg):
    """Reproduce the CLI → config override logic from train.py."""
    if "msp" not in cfg:
        cfg["msp"] = {}
    if args.msp:
        cfg["msp"]["enabled"] = True
    if args.msp_lambda_mask is not None:
        cfg["msp"]["lambda_mask"] = args.msp_lambda_mask
    if args.msp_lambda_ctx is not None:
        cfg["msp"]["lambda_ctx"] = args.msp_lambda_ctx
    if args.msp_mask_ratio is not None:
        cfg["msp"]["mask_ratio"] = args.msp_mask_ratio
    return cfg


class TestAdjustMspCtxLambda:
    """Tests for the progressive warmup of lambda_ctx."""

    @staticmethod
    def _cfg(lambda_ctx=0.5, warmup=None):
        return {"msp": {"lambda_ctx": lambda_ctx, "ctx_warmup_epochs": warmup}}

    def test_no_warmup_returns_base(self):
        cfg = self._cfg(lambda_ctx=0.5, warmup=None)
        assert adjust_msp_ctx_lambda(0, cfg) == 0.5
        assert adjust_msp_ctx_lambda(500, cfg) == 0.5

    def test_zero_lambda_always_zero(self):
        cfg = self._cfg(lambda_ctx=0.0, warmup=[50, 150])
        assert adjust_msp_ctx_lambda(100, cfg) == 0.0

    def test_before_warmup_start(self):
        cfg = self._cfg(lambda_ctx=0.5, warmup=[50, 150])
        assert adjust_msp_ctx_lambda(0, cfg) == 0.0
        assert adjust_msp_ctx_lambda(49.9, cfg) == 0.0

    def test_at_warmup_start(self):
        cfg = self._cfg(lambda_ctx=0.5, warmup=[50, 150])
        assert adjust_msp_ctx_lambda(50, cfg) == 0.0

    def test_mid_warmup(self):
        cfg = self._cfg(lambda_ctx=0.5, warmup=[50, 150])
        val = adjust_msp_ctx_lambda(100, cfg)
        assert abs(val - 0.25) < 1e-6, f"Expected 0.25, got {val}"

    def test_at_warmup_end(self):
        cfg = self._cfg(lambda_ctx=0.5, warmup=[50, 150])
        assert adjust_msp_ctx_lambda(150, cfg) == 0.5

    def test_after_warmup_end(self):
        cfg = self._cfg(lambda_ctx=0.5, warmup=[50, 150])
        assert adjust_msp_ctx_lambda(200, cfg) == 0.5
        assert adjust_msp_ctx_lambda(3000, cfg) == 0.5

    def test_warmup_is_linear(self):
        """Verify linearity at several points."""
        cfg = self._cfg(lambda_ctx=1.0, warmup=[0, 100])
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            epoch = frac * 100
            expected = 1.0 * frac
            actual = adjust_msp_ctx_lambda(epoch, cfg)
            assert abs(actual - expected) < 1e-6, \
                f"Epoch {epoch}: expected {expected}, got {actual}"

    def test_fractional_epoch(self):
        """Sub-epoch granularity (e.g. e + i/iters_per_epoch)."""
        cfg = self._cfg(lambda_ctx=0.5, warmup=[50, 150])
        val = adjust_msp_ctx_lambda(75.5, cfg)
        expected = 0.5 * (75.5 - 50) / 100
        assert abs(val - expected) < 1e-6

    def test_missing_msp_section(self):
        cfg = {}
        assert adjust_msp_ctx_lambda(100, cfg) == 0.0

    def test_missing_lambda_ctx(self):
        cfg = {"msp": {"ctx_warmup_epochs": [50, 150]}}
        assert adjust_msp_ctx_lambda(100, cfg) == 0.0

    def test_same_start_end(self):
        """Edge case: warmup_start == warmup_end → immediate jump."""
        cfg = self._cfg(lambda_ctx=0.5, warmup=[100, 100])
        assert adjust_msp_ctx_lambda(99, cfg) == 0.0
        assert adjust_msp_ctx_lambda(100, cfg) == 0.5


class TestMSPCLIParsing:
    """Tests for MSP CLI argument parsing."""

    def test_all_flags(self):
        args = _parse_msp_args([
            "--msp",
            "--msp-lambda-mask", "2.0",
            "--msp-lambda-ctx", "0.5",
            "--msp-mask-ratio", "0.2", "0.6",
        ])
        assert args.msp is True
        assert args.msp_lambda_mask == 2.0
        assert args.msp_lambda_ctx == 0.5
        assert args.msp_mask_ratio == [0.2, 0.6]

    def test_no_flags_defaults(self):
        args = _parse_msp_args([])
        assert args.msp is False
        assert args.msp_lambda_mask is None
        assert args.msp_lambda_ctx is None
        assert args.msp_mask_ratio is None

    def test_msp_only(self):
        args = _parse_msp_args(["--msp"])
        assert args.msp is True
        assert args.msp_lambda_mask is None

    def test_lambda_without_msp_flag(self):
        """Setting lambda values without --msp should still parse."""
        args = _parse_msp_args(["--msp-lambda-mask", "0.5"])
        assert args.msp is False
        assert args.msp_lambda_mask == 0.5

    def test_mask_ratio_requires_two_values(self):
        with pytest.raises(SystemExit):
            _parse_msp_args(["--msp-mask-ratio", "0.3"])


class TestMSPConfigOverrides:
    """Tests for CLI → config override logic."""

    def test_cli_enables_msp(self):
        args = _parse_msp_args(["--msp"])
        cfg = {"msp": {"enabled": False}}
        cfg = _apply_msp_overrides(args, cfg)
        assert cfg["msp"]["enabled"] is True

    def test_cli_overrides_lambdas(self):
        args = _parse_msp_args([
            "--msp-lambda-mask", "3.0",
            "--msp-lambda-ctx", "0.7",
        ])
        cfg = {"msp": {"lambda_mask": 1.0, "lambda_ctx": 0.0}}
        cfg = _apply_msp_overrides(args, cfg)
        assert cfg["msp"]["lambda_mask"] == 3.0
        assert cfg["msp"]["lambda_ctx"] == 0.7

    def test_cli_overrides_mask_ratio(self):
        args = _parse_msp_args(["--msp-mask-ratio", "0.2", "0.6"])
        cfg = {"msp": {"mask_ratio": [0.3, 0.5]}}
        cfg = _apply_msp_overrides(args, cfg)
        assert cfg["msp"]["mask_ratio"] == [0.2, 0.6]

    def test_no_cli_preserves_config(self):
        args = _parse_msp_args([])
        cfg = {
            "msp": {
                "enabled": True,
                "lambda_mask": 2.0,
                "lambda_ctx": 0.3,
                "mask_ratio": [0.4, 0.6],
            }
        }
        original = dict(cfg["msp"])
        cfg = _apply_msp_overrides(args, cfg)
        assert cfg["msp"] == original

    def test_creates_msp_section_if_missing(self):
        args = _parse_msp_args(["--msp"])
        cfg = {}
        cfg = _apply_msp_overrides(args, cfg)
        assert "msp" in cfg
        assert cfg["msp"]["enabled"] is True

    def test_msp_ctx_enabled_flag(self):
        """msp_ctx_enabled should be True only when both msp_enabled AND lambda_ctx > 0."""
        cfg = {"msp": {"enabled": True, "lambda_ctx": 0.5}}
        msp_enabled = cfg["msp"].get("enabled", False)
        msp_ctx_enabled = msp_enabled and cfg["msp"].get("lambda_ctx", 0.0) > 0
        assert msp_ctx_enabled is True

        cfg2 = {"msp": {"enabled": True, "lambda_ctx": 0.0}}
        msp_ctx_enabled2 = cfg2["msp"]["enabled"] and cfg2["msp"]["lambda_ctx"] > 0
        assert msp_ctx_enabled2 is False

        cfg3 = {"msp": {"enabled": False, "lambda_ctx": 0.5}}
        msp_ctx_enabled3 = cfg3["msp"]["enabled"] and cfg3["msp"]["lambda_ctx"] > 0
        assert msp_ctx_enabled3 is False

class TestMSPYAMLConfig:
    """Tests for YAML config parsing of MSP fields."""

    @pytest.fixture
    def config_path(self):
        return os.path.join(
            os.path.dirname(__file__),
            "../../med_slim/configs/pretrain.yml",
        )

    def test_msp_section_exists(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "msp" in cfg

    def test_msp_default_fields(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        msp = cfg["msp"]
        assert "enabled" in msp
        assert "lambda_mask" in msp
        assert "lambda_ctx" in msp
        assert "mask_ratio" in msp
        assert "predictor_depth" in msp
        assert "max_seq_len" in msp

    def test_msp_new_fields(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        msp = cfg["msp"]
        assert "ctx_distance_weighted" in msp
        assert isinstance(msp["ctx_distance_weighted"], bool)

    def test_mask_ratio_is_list(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg["msp"]["mask_ratio"], list)
        assert len(cfg["msp"]["mask_ratio"]) == 2

    def test_ctx_warmup_epochs_nullable(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        warmup = cfg["msp"].get("ctx_warmup_epochs")
        assert warmup is None or (isinstance(warmup, list) and len(warmup) == 2)

    def test_round_trip(self, config_path, tmp_path):
        """YAML dump → load preserves MSP config."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        out = tmp_path / "config.yml"
        with open(out, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        with open(out) as f:
            cfg2 = yaml.safe_load(f)
        assert cfg["msp"] == cfg2["msp"]
