"""
Integration tests for loading a COBRA model from a real pretrained checkpoint.

Uses the checkpoint at:
    /hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-03-15-04:23/medslim-epoch2000.pth.tar

Checkpoint details (mamba2 + abmil):
    input_dims: [512, 768, 1024, 1152, 1376, 1536]
    embed_dim:  1024
    contrast_dim: 256
    num_heads:  8
    num_layers: 2
    mamba_d_state: 128
    attn_dim:   256
"""

import os
import torch
import yaml
import pytest
from accelerate import Accelerator

from med_slim.eval.load_cobra import load_pretrained_cobra, _build_cobra
from med_slim.model.sequence_encoder.cobra import Cobra


CKPT_DIR = "/hpcwork/rwth1833/checkpoints/MedSliM-pretraining/MRNet-fastMRI-KMAR50K/2026-03-15-04:23"
CKPT_PATH = os.path.join(CKPT_DIR, "medslim-epoch2000.pth.tar")
CONFIG_PATH = os.path.join(CKPT_DIR, "config.yaml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

skip_no_ckpt = pytest.mark.skipif(
    not os.path.isfile(CKPT_PATH),
    reason=f"Checkpoint not found: {CKPT_PATH}",
)


@pytest.fixture(scope="module")
def training_config():
    """Load the saved training config YAML."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def model_config(training_config):
    """Extract the model.cobra section used by load_pretrained_cobra."""
    return training_config["model"]["cobra"]


@pytest.fixture(scope="module")
def accelerator():
    return Accelerator(cpu=not torch.cuda.is_available())


@pytest.fixture(scope="module")
def checkpoint():
    """Load the raw checkpoint once for the entire module."""
    return torch.load(CKPT_PATH, map_location="cpu", weights_only=False)


# Checkpoint structure tests
@skip_no_ckpt
class TestCheckpointStructure:
    """Verify the checkpoint contains expected metadata and state dict layout."""

    def test_top_level_keys(self, checkpoint):
        required = {"epoch", "state_dict", "optimizer", "sequence_encoder", "pooling"}
        assert required.issubset(checkpoint.keys()), (
            f"Missing keys: {required - checkpoint.keys()}"
        )

    def test_epoch_value(self, checkpoint):
        assert checkpoint["epoch"] == 2000

    def test_sequence_encoder_metadata(self, checkpoint):
        assert checkpoint["sequence_encoder"] == "mamba2"

    def test_pooling_metadata(self, checkpoint):
        assert checkpoint["pooling"] == "abmil"

    def test_state_dict_has_both_encoders(self, checkpoint):
        keys = list(checkpoint["state_dict"].keys())
        has_base = any(k.startswith("base_encoder.") for k in keys)
        has_momentum = any(k.startswith("momentum_encoder.") for k in keys)
        assert has_base, "No base_encoder weights in state_dict"
        assert has_momentum, "No momentum_encoder weights in state_dict"

    def test_no_legacy_mamba_enc_keys(self, checkpoint):
        """Confirm no legacy 'mamba_enc' keys exist (should be 'seq_enc')."""
        for k in checkpoint["state_dict"].keys():
            assert "mamba_enc" not in k, f"Legacy key found: {k}"


# Loading tests (momentum encoder, default)
@skip_no_ckpt
class TestLoadMomentumEncoder:
    """Load the momentum encoder from checkpoint and verify correctness."""

    @pytest.fixture(scope="class")
    def cobra(self, model_config, accelerator):
        return load_pretrained_cobra(
            checkpoint_path=CKPT_PATH,
            accelerator=accelerator,
            model_config=model_config,
            encoder_type="momentum",
        )

    def test_returns_cobra_model(self, cobra):
        assert isinstance(cobra, Cobra)

    def test_model_in_inference_mode(self, cobra):
        assert cobra.mode == "inference"

    def test_sequence_encoder_type(self, cobra):
        assert cobra.sequence_encoder == "mamba2"

    def test_slice_pooling_type(self, cobra):
        assert cobra.slice_pooling == "abmil"

    def test_embedding_layers_present(self, cobra, model_config):
        expected_dims = model_config["input_dims"]
        for d in expected_dims:
            assert str(d) in cobra.embed, f"Missing embed layer for dim={d}"

    def test_forward_single_fm(self, cobra):
        """Forward pass with a single FM embedding dimension (768 = RAD-DINO).
        In inference mode, x must be a list of tensors (one per FM)."""
        cobra_eval = cobra.to(DEVICE).eval()
        x = [torch.randn(2, 20, 768, device=DEVICE)]
        with torch.no_grad():
            out = cobra_eval(x)
        assert out.shape[0] == 2
        assert torch.isfinite(out).all()

    def test_forward_multi_fm(self, cobra, model_config):
        """Forward pass with all FM embedding dimensions simultaneously."""
        cobra_eval = cobra.to(DEVICE).eval()
        num_slices = 16
        x = [torch.randn(1, num_slices, dim, device=DEVICE) for dim in model_config["input_dims"]]
        with torch.no_grad():
            out = cobra_eval(x)
        assert torch.isfinite(out).all()

    def test_forward_each_fm_dim_individually(self, cobra, model_config):
        """Forward pass with each supported FM embedding dimension individually."""
        cobra_eval = cobra.to(DEVICE).eval()
        for dim in model_config["input_dims"]:
            x = [torch.randn(1, 16, dim, device=DEVICE)]
            with torch.no_grad():
                out = cobra_eval(x)
            assert torch.isfinite(out).all(), f"Non-finite output for input_dim={dim}"

    def test_forward_with_seq_lengths(self, cobra):
        """Variable-length sequences with padding should work correctly."""
        cobra_eval = cobra.to(DEVICE).eval()
        batch_size, max_slices = 4, 32
        x = [torch.randn(batch_size, max_slices, 1024, device=DEVICE)]
        seq_lengths = torch.tensor([10, 20, 32, 15], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = cobra_eval(x, seq_lengths=seq_lengths)
        assert out.shape[0] == batch_size
        assert torch.isfinite(out).all()

    def test_attention_extraction(self, cobra):
        """ABMIL attention weights should be extractable and well-formed."""
        cobra_eval = cobra.to(DEVICE).eval()
        batch_size, num_slices = 2, 16
        x = [torch.randn(batch_size, num_slices, 768, device=DEVICE)]
        with torch.no_grad():
            attn = cobra_eval(x, get_attention=True)
        assert attn.shape == (batch_size, 1, num_slices)
        assert torch.isfinite(attn).all()
        assert (attn >= 0).all()

    def test_attention_sums_to_one(self, cobra):
        """Attention weights over valid slices should sum to ~1."""
        cobra_eval = cobra.to(DEVICE).eval()
        x = [torch.randn(2, 10, 768, device=DEVICE)]
        with torch.no_grad():
            attn = cobra_eval(x, get_attention=True)
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_attention_masked_positions_zero(self, cobra):
        """Padded positions should receive zero attention."""
        cobra_eval = cobra.to(DEVICE).eval()
        batch_size, max_slices = 2, 20
        x = [torch.randn(batch_size, max_slices, 768, device=DEVICE)]
        seq_lengths = torch.tensor([8, 15], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            attn = cobra_eval(x, seq_lengths=seq_lengths, get_attention=True)
        assert torch.allclose(
            attn[0, 0, 8:], torch.zeros(max_slices - 8, device=DEVICE), atol=1e-6
        )
        assert torch.allclose(
            attn[1, 0, 15:], torch.zeros(max_slices - 15, device=DEVICE), atol=1e-6
        )


# Loading tests (base encoder)
@skip_no_ckpt
class TestLoadBaseEncoder:
    """Load the base (online) encoder and verify it differs from momentum."""

    @pytest.fixture(scope="class")
    def cobra_base(self, model_config, accelerator):
        return load_pretrained_cobra(
            checkpoint_path=CKPT_PATH,
            accelerator=accelerator,
            model_config=model_config,
            encoder_type="base",
        )

    @pytest.fixture(scope="class")
    def cobra_momentum(self, model_config, accelerator):
        return load_pretrained_cobra(
            checkpoint_path=CKPT_PATH,
            accelerator=accelerator,
            model_config=model_config,
            encoder_type="momentum",
        )

    def test_base_loads_successfully(self, cobra_base):
        assert isinstance(cobra_base, Cobra)

    def test_base_forward_pass(self, cobra_base):
        cobra_eval = cobra_base.to(DEVICE).eval()
        x = [torch.randn(2, 16, 768, device=DEVICE)]
        with torch.no_grad():
            out = cobra_eval(x)
        assert torch.isfinite(out).all()

    def test_base_vs_momentum_weights_differ(self, cobra_base, cobra_momentum):
        """After 2000 epochs the base and momentum weights should have diverged."""
        base_params = {n: p.cpu() for n, p in cobra_base.named_parameters()}
        for name, m_param in cobra_momentum.named_parameters():
            if name in base_params:
                if not torch.allclose(base_params[name], m_param.cpu(), atol=1e-7):
                    return  # found at least one difference
        pytest.fail("All parameters are identical between base and momentum encoders")


# Weight-level integrity tests
@skip_no_ckpt
class TestWeightIntegrity:
    """Verify that loaded weights match the raw checkpoint values exactly."""

    def test_momentum_weights_match_checkpoint(self, model_config, accelerator, checkpoint):
        model = load_pretrained_cobra(
            checkpoint_path=CKPT_PATH,
            accelerator=accelerator,
            model_config=model_config,
            encoder_type="momentum",
        )
        chkpt = checkpoint["state_dict"]
        prefix = "momentum_encoder."
        proj_prefix = "momentum_encoder.proj"

        model_sd = model.state_dict()
        for k, v in chkpt.items():
            if k.startswith(prefix) and not k.startswith(proj_prefix):
                clean_key = k[len(prefix):]
                if clean_key in model_sd:
                    assert torch.equal(model_sd[clean_key].cpu(), v.cpu()), (
                        f"Weight mismatch for {clean_key}"
                    )

    def test_no_unexpected_keys(self, model_config, accelerator, checkpoint):
        """All loaded keys should exist in the model's state dict (ignoring proj)."""
        model = _build_cobra(
            model_config, sequence_encoder="mamba2", slice_pooling="abmil",
            fm_pooling="avg_pool",
        )
        chkpt = checkpoint["state_dict"]
        prefix = "momentum_encoder."
        proj_prefix = "momentum_encoder.proj"

        model_keys = set(model.state_dict().keys())
        for k in chkpt:
            if k.startswith(prefix) and not k.startswith(proj_prefix):
                clean_key = k[len(prefix):]
                assert clean_key in model_keys, f"Unexpected key in checkpoint: {clean_key}"


# Edge cases and error handling
class TestLoadCobraErrors:
    """Test error handling paths in load_pretrained_cobra."""

    def test_missing_checkpoint_raises_file_not_found(self, accelerator):
        with pytest.raises(FileNotFoundError):
            load_pretrained_cobra(
                checkpoint_path="/nonexistent/path.pth.tar",
                accelerator=accelerator,
                model_config={"embed_dim": 768, "input_dims": [768], "contrast_dim": 256,
                              "num_heads": 4, "num_layers": 1, "dropout": 0.1,
                              "mamba_d_state": 64, "attn_dim": 128},
            )

    @skip_no_ckpt
    def test_invalid_encoder_type_raises(self, model_config, accelerator):
        """Requesting weights for a nonexistent encoder prefix should raise ValueError."""
        with pytest.raises(ValueError, match="No .* encoder weights found"):
            load_pretrained_cobra(
                checkpoint_path=CKPT_PATH,
                accelerator=accelerator,
                model_config=model_config,
                encoder_type="nonexistent",
            )
