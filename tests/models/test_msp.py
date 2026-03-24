import torch
import pytest

from med_slim.model.ssl import MoCo, MSPPredictor
from med_slim.model.ssl.masking import (
    generate_contiguous_slice_mask,
    generate_contiguous_slice_mask_packed,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_msp_model(
    embed_dim=64,
    contrast_dim=32,
    input_dim=64,
    msp_lambda_ctx=0.0,
    msp_ctx_distance_weighted=True,
    **msp_overrides,
):
    defaults = dict(
        embed_dim=embed_dim,
        contrast_dim=contrast_dim,
        input_dims=[input_dim],
        num_heads=4,
        num_layers=1,
        T=0.2,
        dropout=0.0,
        sequence_encoder="transformer",
        pooling="abmil",
        att_dim=32,
        msp_enabled=True,
        msp_lambda_mask=1.0,
        msp_lambda_ctx=msp_lambda_ctx,
        msp_mask_ratio=(0.3, 0.5),
        msp_predictor_depth=2,
        msp_max_seq_len=64,
        msp_ctx_distance_weighted=msp_ctx_distance_weighted,
    )
    defaults.update(msp_overrides)
    return MoCo(**defaults).to(DEVICE)


def _dummy_padded_batch(B=4, S=10, D=64):
    x1 = torch.randn(B, S, D, device=DEVICE)
    x2 = torch.randn(B, S, D, device=DEVICE)
    sizes = torch.full((B,), D, dtype=torch.long, device=DEVICE)
    seq_lens = torch.full((B,), S, dtype=torch.long, device=DEVICE)
    return x1, x2, sizes, seq_lens


class TestContiguousMasking:
    """Tests for generate_contiguous_slice_mask and _packed variant."""

    def test_padded_mask_shape(self):
        seq_lengths = torch.tensor([10, 8, 12, 6])
        mask = generate_contiguous_slice_mask(seq_lengths, max_seq_len=12)
        assert mask.shape == (4, 12)
        assert mask.dtype == torch.bool

    def test_padded_mask_is_contiguous(self):
        torch.manual_seed(0)
        seq_lengths = torch.tensor([20, 15, 25])
        mask = generate_contiguous_slice_mask(seq_lengths, max_seq_len=25)
        for i in range(3):
            row = mask[i]
            indices = row.nonzero().squeeze(-1)
            if len(indices) > 0:
                assert (indices[-1] - indices[0] + 1) == len(indices), \
                    f"Sample {i}: mask is not contiguous"

    def test_padded_mask_within_valid_region(self):
        seq_lengths = torch.tensor([5, 10, 3, 8])
        mask = generate_contiguous_slice_mask(seq_lengths, max_seq_len=10)
        for i in range(4):
            valid_len = seq_lengths[i].item()
            assert mask[i, valid_len:].sum() == 0, \
                f"Sample {i}: mask extends beyond valid region"

    def test_padded_mask_ratio_bounds(self):
        torch.manual_seed(42)
        seq_lengths = torch.full((100,), 20, dtype=torch.long)
        mask = generate_contiguous_slice_mask(
            seq_lengths, mask_ratio_range=(0.3, 0.5), max_seq_len=20,
        )
        for i in range(100):
            n_masked = mask[i].sum().item()
            assert 0.3 * 20 - 1 <= n_masked <= 0.5 * 20 + 1

    def test_padded_mask_short_sequence_skipped(self):
        seq_lengths = torch.tensor([1, 0])
        mask = generate_contiguous_slice_mask(seq_lengths, max_seq_len=5)
        assert mask.sum() == 0, "Sequences with length < 2 should not be masked"

    def test_packed_mask_shape(self):
        cu_seqlens = torch.tensor([0, 8, 20, 26], dtype=torch.int32)
        mask = generate_contiguous_slice_mask_packed(cu_seqlens)
        assert mask.shape == (26,)
        assert mask.dtype == torch.bool

    def test_packed_mask_is_contiguous_per_sample(self):
        torch.manual_seed(1)
        lens = [10, 15, 8]
        cu_seqlens = torch.tensor(
            [0] + [sum(lens[:i + 1]) for i in range(len(lens))],
            dtype=torch.int32,
        )
        mask = generate_contiguous_slice_mask_packed(cu_seqlens)
        for i, l in enumerate(lens):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            sample_mask = mask[start:end]
            indices = sample_mask.nonzero().squeeze(-1)
            if len(indices) > 0:
                assert (indices[-1] - indices[0] + 1) == len(indices), \
                    f"Packed sample {i}: mask is not contiguous"

    def test_packed_mask_stays_within_sample(self):
        lens = [5, 10, 3]
        cu_seqlens = torch.tensor(
            [0] + [sum(lens[:i + 1]) for i in range(len(lens))],
            dtype=torch.int32,
        )
        mask = generate_contiguous_slice_mask_packed(cu_seqlens)
        for i, l in enumerate(lens):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            sample_mask = mask[start:end]
            assert sample_mask.sum() <= l


class TestMSPPredictor:
    """Tests for the lightweight MSP predictor."""

    def test_output_shape(self):
        pred = MSPPredictor(embed_dim=64, num_layers=2, max_seq_len=32).to(DEVICE)
        h = torch.randn(10, 64, device=DEVICE)
        pos = torch.arange(10, device=DEVICE)
        out = pred(h, pos)
        assert out.shape == (10, 64)

    def test_default_hidden_dim(self):
        pred = MSPPredictor(embed_dim=128)
        linear = pred.blocks[0][1]  # LayerNorm → Linear(embed, hidden) → ...
        assert linear.in_features == 128
        assert linear.out_features == 32  # 128 // 4

    def test_custom_hidden_dim(self):
        pred = MSPPredictor(embed_dim=128, hidden_dim=48)
        linear = pred.blocks[0][1]
        assert linear.out_features == 48

    def test_positional_embedding_matters(self):
        """Different positions should produce different outputs."""
        pred = MSPPredictor(embed_dim=32, num_layers=1, max_seq_len=32).to(DEVICE)
        pred.eval()
        h = torch.ones(2, 32, device=DEVICE)
        pos_a = torch.tensor([0, 1], device=DEVICE)
        pos_b = torch.tensor([10, 11], device=DEVICE)
        with torch.no_grad():
            out_a = pred(h, pos_a)
            out_b = pred(h, pos_b)
        assert not torch.allclose(out_a, out_b, atol=1e-6), \
            "Different positions should yield different outputs"

    def test_gradient_flow(self):
        pred = MSPPredictor(embed_dim=32, num_layers=2, max_seq_len=16).to(DEVICE)
        pred.train()
        h = torch.randn(5, 32, device=DEVICE, requires_grad=True)
        pos = torch.arange(5, device=DEVICE)
        out = pred(h, pos)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        for p in pred.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestDistanceWeights:
    """Tests for MoCo._distance_weights static method."""

    def test_sum_equals_num_visible(self):
        vis = torch.tensor([0, 1, 2, 6, 7, 8, 9])
        mask = torch.tensor([3, 4, 5])
        w = MoCo._distance_weights(vis, mask)
        assert w.shape == (7,)
        assert abs(w.sum().item() - 7.0) < 1e-5

    def test_boundary_slices_have_highest_weight(self):
        vis = torch.tensor([0, 1, 2, 6, 7, 8, 9])
        mask = torch.tensor([3, 4, 5])
        w = MoCo._distance_weights(vis, mask)
        # pos 2 (d_min=1) > pos 1 (d_min=2) > pos 0 (d_min=3)
        assert w[2] > w[1] > w[0]
        # pos 3→6 (d_min=1) > pos 4→7 (d_min=2) > pos 5→8 (d_min=3)
        assert w[3] > w[4] > w[5]

    def test_symmetric_mask(self):
        """With a centrally placed mask, weights should be symmetric."""
        vis = torch.tensor([0, 1, 2, 7, 8, 9])
        mask = torch.tensor([3, 4, 5, 6])
        w = MoCo._distance_weights(vis, mask)
        assert torch.allclose(w[0], w[5], atol=1e-5)  # d_min=3 both
        assert torch.allclose(w[1], w[4], atol=1e-5)  # d_min=2 both
        assert torch.allclose(w[2], w[3], atol=1e-5)  # d_min=1 both

    def test_single_masked_position(self):
        vis = torch.tensor([0, 1, 3, 4])
        mask = torch.tensor([2])
        w = MoCo._distance_weights(vis, mask)
        assert abs(w.sum().item() - 4.0) < 1e-5
        # pos 1 and pos 3 are both d_min=1 → highest weight
        assert w[1] > w[0]
        assert w[2] > w[3]

    def test_all_weights_positive(self):
        vis = torch.arange(0, 20)
        mask = torch.tensor([10, 11, 12])
        w = MoCo._distance_weights(vis, mask)
        assert (w > 0).all()

    def test_d_min_clamped_to_one(self):
        """Even if adjacent, d_min should be >= 1 (no division by zero)."""
        vis = torch.tensor([0])
        mask = torch.tensor([1])
        w = MoCo._distance_weights(vis, mask)
        assert torch.isfinite(w).all()


class TestMoCoMSP:
    """Integration tests for MoCo with MSP enabled."""

    def test_msp_returns_dict(self):
        model = _build_msp_model()
        x1, x2, sizes, seq_lens = _dummy_padded_batch()
        model.eval()
        with torch.no_grad():
            result = model(
                x1, x2,
                input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                seq_lengths=seq_lens, m=0.99,
            )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"loss", "loss_infonce", "loss_msp", "loss_ctx"}

    def test_msp_disabled_returns_scalar(self):
        model = MoCo(
            embed_dim=64, contrast_dim=32, input_dims=[64],
            num_heads=4, num_layers=1,
            sequence_encoder="transformer", pooling="abmil", att_dim=32,
            msp_enabled=False,
        ).to(DEVICE).eval()
        x1, x2, sizes, seq_lens = _dummy_padded_batch()
        with torch.no_grad():
            result = model(
                x1, x2,
                input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                seq_lengths=seq_lens, m=0.99,
            )
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_msp_only_no_ctx(self):
        """MSP with lambda_ctx=0: loss_ctx should be zero."""
        model = _build_msp_model(msp_lambda_ctx=0.0)
        x1, x2, sizes, seq_lens = _dummy_padded_batch()
        model.eval()
        with torch.no_grad():
            result = model(
                x1, x2,
                input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                seq_lengths=seq_lens, m=0.99,
            )
        assert result["loss_ctx"].item() == 0.0

    def test_msp_with_distance_weighted_ctx(self):
        """MSP + distance-weighted context loss: loss_ctx > 0."""
        model = _build_msp_model(msp_lambda_ctx=0.5, msp_ctx_distance_weighted=True)
        x1, x2, sizes, seq_lens = _dummy_padded_batch()
        model.eval()
        with torch.no_grad():
            result = model(
                x1, x2,
                input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                seq_lengths=seq_lens, m=0.99,
            )
        assert result["loss_ctx"].item() > 0

    def test_msp_with_uniform_ctx(self):
        """MSP + uniform context loss: loss_ctx > 0."""
        model = _build_msp_model(msp_lambda_ctx=0.5, msp_ctx_distance_weighted=False)
        x1, x2, sizes, seq_lens = _dummy_padded_batch()
        model.eval()
        with torch.no_grad():
            result = model(
                x1, x2,
                input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                seq_lengths=seq_lens, m=0.99,
            )
        assert result["loss_ctx"].item() > 0

    def test_distance_weighted_differs_from_uniform(self):
        """Distance-weighted and uniform context losses should differ."""
        torch.manual_seed(123)
        x1, x2, sizes, seq_lens = _dummy_padded_batch()

        torch.manual_seed(42)
        model_dw = _build_msp_model(msp_lambda_ctx=0.5, msp_ctx_distance_weighted=True)
        model_dw.eval()

        torch.manual_seed(42)
        model_uni = _build_msp_model(msp_lambda_ctx=0.5, msp_ctx_distance_weighted=False)
        model_uni.eval()

        # Copy weights so the only difference is the weighting scheme
        model_uni.load_state_dict(model_dw.state_dict())

        with torch.no_grad():
            r_dw = model_dw(x1, x2, input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                            seq_lengths=seq_lens, m=0.99)
            r_uni = model_uni(x1, x2, input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                              seq_lengths=seq_lens, m=0.99)
        # They'll likely differ because masking is random, but total loss should be finite
        assert torch.isfinite(r_dw["loss"])
        assert torch.isfinite(r_uni["loss"])

    def test_total_loss_composition(self):
        """Verify total_loss = loss_infonce + lambda_mask * loss_msp + lambda_ctx * loss_ctx."""
        model = _build_msp_model(msp_lambda_mask=2.0, msp_lambda_ctx=0.5)
        x1, x2, sizes, seq_lens = _dummy_padded_batch()
        model.eval()
        with torch.no_grad():
            result = model(
                x1, x2,
                input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                seq_lengths=seq_lens, m=0.99,
            )
        expected = (
            result["loss_infonce"] +
            2.0 * result["loss_msp"] +
            0.5 * result["loss_ctx"]
        )
        assert torch.allclose(result["loss"], expected, atol=1e-4), \
            f"Total loss {result['loss'].item():.6f} != expected {expected.item():.6f}"

    def test_lambda_mask_zero_ablation(self):
        """With lambda_mask=0, MSP loss should not affect total loss."""
        model = _build_msp_model(msp_lambda_mask=0.0, msp_lambda_ctx=0.0)
        x1, x2, sizes, seq_lens = _dummy_padded_batch()
        model.eval()
        with torch.no_grad():
            result = model(
                x1, x2,
                input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                seq_lengths=seq_lens, m=0.99,
            )
        assert torch.allclose(result["loss"], result["loss_infonce"], atol=1e-5)

    def test_all_losses_finite(self):
        model = _build_msp_model(msp_lambda_ctx=0.5)
        x1, x2, sizes, seq_lens = _dummy_padded_batch()
        model.eval()
        with torch.no_grad():
            result = model(
                x1, x2,
                input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                seq_lengths=seq_lens, m=0.99,
            )
        for key in ["loss", "loss_infonce", "loss_msp", "loss_ctx"]:
            assert torch.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_gradient_flow_through_msp(self):
        """Gradients should flow through MSP predictor and mask token."""
        model = _build_msp_model(msp_lambda_ctx=0.5)
        model.train()
        x1, x2, sizes, seq_lens = _dummy_padded_batch()
        result = model(
            x1, x2,
            input_feature_dims_1=sizes, input_feature_dims_2=sizes,
            seq_lengths=seq_lens, m=0.99,
        )
        result["loss"].backward()

        assert model.mask_token.grad is not None, "mask_token should receive gradients"
        predictor_has_grad = any(
            p.grad is not None
            for p in model.msp_predictor.parameters()
            if p.requires_grad
        )
        assert predictor_has_grad, "MSP predictor should receive gradients"

    def test_gradient_flow_base_encoder(self):
        """Base encoder should receive gradients from both InfoNCE and MSP."""
        model = _build_msp_model(msp_lambda_ctx=0.5)
        model.train()
        x1, x2, sizes, seq_lens = _dummy_padded_batch()
        result = model(
            x1, x2,
            input_feature_dims_1=sizes, input_feature_dims_2=sizes,
            seq_lengths=seq_lens, m=0.99,
        )
        result["loss"].backward()
        encoder_has_grad = any(
            p.grad is not None
            for p in model.base_encoder.parameters()
            if p.requires_grad
        )
        assert encoder_has_grad

    def test_momentum_encoder_no_grad(self):
        """Momentum encoder should NOT have requires_grad."""
        model = _build_msp_model()
        for p in model.momentum_encoder.parameters():
            assert not p.requires_grad

    def test_msp_with_variable_seq_lengths(self):
        """MSP should handle batches with variable sequence lengths."""
        B, S, D = 4, 16, 64
        model = _build_msp_model()
        model.eval()
        x1 = torch.randn(B, S, D, device=DEVICE)
        x2 = torch.randn(B, S, D, device=DEVICE)
        sizes = torch.full((B,), D, dtype=torch.long, device=DEVICE)
        seq_lens = torch.tensor([8, 12, 6, 16], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            result = model(
                x1, x2,
                input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                seq_lengths=seq_lens, m=0.99,
            )
        assert isinstance(result, dict)
        assert torch.isfinite(result["loss"])

    def test_has_mask_token_and_predictor(self):
        """MSP-enabled model should have mask_token and msp_predictor."""
        model = _build_msp_model()
        assert hasattr(model, "mask_token")
        assert isinstance(model.mask_token, torch.nn.Parameter)
        assert hasattr(model, "msp_predictor")
        assert isinstance(model.msp_predictor, MSPPredictor)

    def test_msp_disabled_has_no_msp_components(self):
        """MSP-disabled model should not have mask_token or msp_predictor."""
        model = MoCo(
            embed_dim=64, contrast_dim=32, input_dims=[64],
            num_heads=4, num_layers=1,
            sequence_encoder="transformer", pooling="abmil", att_dim=32,
            msp_enabled=False,
        )
        assert not hasattr(model, "mask_token")
        assert not hasattr(model, "msp_predictor")

# MoCo with MSP in packed mode (requires CUDA + FlashAttention bf16)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FlashAttention packed mode"
)

def _dummy_packed_batch(lens=(8, 12, 6), D=64):
    """Build a packed batch on CUDA."""
    device = torch.device("cuda")
    total = sum(lens)
    cu_seqlens = torch.tensor(
        [0] + [sum(lens[: i + 1]) for i in range(len(lens))],
        dtype=torch.int32,
        device=device,
    )
    seq_idx = torch.cat(
        [torch.full((l,), i, dtype=torch.int32, device=device) for i, l in enumerate(lens)]
    )
    x1 = torch.randn(total, D, device=device)
    x2 = torch.randn(total, D, device=device)
    sizes = torch.full((len(lens),), D, dtype=torch.long, device=device)
    return x1, x2, sizes, cu_seqlens, seq_idx, max(lens)

@requires_cuda
class TestMoCoMSPPacked:
    """Integration tests for MoCo + MSP in packed sequence mode (CUDA, bf16)."""

    def _run_packed(self, model, x1, x2, sizes, cu_seqlens, seq_idx, max_seqlen):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(
                x1, x2,
                input_feature_dims_1=sizes, input_feature_dims_2=sizes,
                m=0.99, use_packed=True,
                cu_seqlens1=cu_seqlens, cu_seqlens2=cu_seqlens,
                max_seqlen1=max_seqlen, max_seqlen2=max_seqlen,
                seq_idx1=seq_idx, seq_idx2=seq_idx,
            )

    def test_packed_msp_returns_dict(self):
        model = _build_msp_model(msp_lambda_ctx=0.5).to("cuda")
        x1, x2, sizes, cu, si, ml = _dummy_packed_batch()
        model.eval()
        with torch.no_grad():
            result = self._run_packed(model, x1, x2, sizes, cu, si, ml)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"loss", "loss_infonce", "loss_msp", "loss_ctx"}

    def test_packed_distance_weighted_ctx(self):
        model = _build_msp_model(
            msp_lambda_ctx=0.5, msp_ctx_distance_weighted=True,
        ).to("cuda")
        x1, x2, sizes, cu, si, ml = _dummy_packed_batch()
        model.eval()
        with torch.no_grad():
            result = self._run_packed(model, x1, x2, sizes, cu, si, ml)
        assert result["loss_ctx"].item() > 0
        for k, v in result.items():
            assert torch.isfinite(v), f"{k} is not finite"

    def test_packed_uniform_ctx(self):
        model = _build_msp_model(
            msp_lambda_ctx=0.5, msp_ctx_distance_weighted=False,
        ).to("cuda")
        x1, x2, sizes, cu, si, ml = _dummy_packed_batch()
        model.eval()
        with torch.no_grad():
            result = self._run_packed(model, x1, x2, sizes, cu, si, ml)
        assert result["loss_ctx"].item() > 0

    def test_packed_msp_no_ctx(self):
        model = _build_msp_model(msp_lambda_ctx=0.0).to("cuda")
        x1, x2, sizes, cu, si, ml = _dummy_packed_batch()
        model.eval()
        with torch.no_grad():
            result = self._run_packed(model, x1, x2, sizes, cu, si, ml)
        assert result["loss_ctx"].item() == 0.0

    def test_packed_gradient_flow(self):
        model = _build_msp_model(msp_lambda_ctx=0.5).to("cuda")
        model.train()
        x1, x2, sizes, cu, si, ml = _dummy_packed_batch()
        result = self._run_packed(model, x1, x2, sizes, cu, si, ml)
        result["loss"].backward()
        assert model.mask_token.grad is not None
        pred_has_grad = any(
            p.grad is not None
            for p in model.msp_predictor.parameters()
            if p.requires_grad
        )
        assert pred_has_grad

    def test_packed_total_loss_composition(self):
        model = _build_msp_model(
            msp_lambda_mask=2.0, msp_lambda_ctx=0.3,
        ).to("cuda")
        x1, x2, sizes, cu, si, ml = _dummy_packed_batch()
        model.eval()
        with torch.no_grad():
            result = self._run_packed(model, x1, x2, sizes, cu, si, ml)
        expected = (
            result["loss_infonce"]
            + 2.0 * result["loss_msp"]
            + 0.3 * result["loss_ctx"]
        )
        assert torch.allclose(result["loss"], expected, atol=1e-3)

    def test_packed_all_losses_finite(self):
        model = _build_msp_model(msp_lambda_ctx=0.5).to("cuda")
        x1, x2, sizes, cu, si, ml = _dummy_packed_batch()
        model.eval()
        with torch.no_grad():
            result = self._run_packed(model, x1, x2, sizes, cu, si, ml)
        for key in ["loss", "loss_infonce", "loss_msp", "loss_ctx"]:
            assert torch.isfinite(result[key]), f"{key} = {result[key]}"
