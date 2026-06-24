"""Unit tests for the variable-K FM-Router and router auxiliary loss formulas."""
import math
import pytest
import torch

from med_slim.model.sequence_encoder.fm_router import FMRouter
from med_slim.model.ssl import MoCo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _construct_router(num_fms=4, embed_dim=16, **kwargs):
    return FMRouter(embed_dim=embed_dim, num_fms=num_fms, **kwargs).to(DEVICE).eval()


def _create_syn_input(k, batch=2, num_slices=3, embed_dim=16):
    fm_embs = torch.randn(k, batch, num_slices, embed_dim, device=DEVICE)
    fm_ids = torch.tensor([list(range(k))] * batch, dtype=torch.long, device=DEVICE)
    return fm_embs, fm_ids


def test_soft_router():
    router = _construct_router()
    fm_embs, fm_ids = _create_syn_input(k=3)
    fused, stats = router(fm_embs, fm_ids, return_stats=True)
    assert fused.shape == (2, 3, 16)  # [B, D, E]
    assert stats["fm_weights_local"].shape == (2, 3, 3)
    sums = stats["fm_weights_local"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert torch.isfinite(fused).all()


def test_variable_k_router_shares_parameters():
    """Same parameters score K=2 and K=4 without shape errors."""
    router = _construct_router(num_fms=4)
    for k in (2, 4):
        fm_embs, fm_ids = _create_syn_input(k=k)
        fused, stats = router(fm_embs, fm_ids, return_stats=True)
        assert fused.shape == (2, 3, 16)
        assert stats["fm_weights_local"].shape == (2, 3, k)



def test_topk_router_selects_k_experts():
    router = _construct_router(num_fms=4, router_mode="topk", top_k=2)
    fm_embs, fm_ids = _create_syn_input(k=4)
    _, stats = router(fm_embs, fm_ids, return_stats=True)
    w = stats["fm_weights_local"]
    # At most top_k weights are non-zero per slice.
    nonzero = (w > 1e-6).sum(dim=-1)
    assert (nonzero <= 2).all()
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_fm_embedding_distinguishes_duplicate_dim_fms():
    """FM identity embedding lets identical token features route differently by ID."""
    router = _construct_router(num_fms=4, use_fm_embedding=True)
    batch, num_slices, embed_dim = 1, 1, 16
    # Two FMs with identical token features but different global IDs.
    token = torch.randn(1, batch, num_slices, embed_dim, device=DEVICE)
    fm_embs = torch.cat([token, token], dim=0)  # [2, B, D, E]
    ids_a = torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE)
    ids_b = torch.tensor([[2, 3]], dtype=torch.long, device=DEVICE)
    _, stats_a = router(fm_embs, ids_a, return_stats=True)
    _, stats_b = router(fm_embs, ids_b, return_stats=True)
    # Different FM IDs should generally yield different routing weights.
    assert not torch.allclose(
        stats_a["fm_weights_local"], stats_b["fm_weights_local"], atol=1e-4
    )


def test_fm_embedding_does_not_leak_into_fused_values():
    """Uniform routing should reduce to avg-pool even when scorer sees FM IDs."""
    router = _construct_router(num_fms=4, use_fm_embedding=True, use_fm_logit_bias=False)
    fm_embs, fm_ids = _create_syn_input(k=4)

    for module in router.scorer.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)
            torch.nn.init.zeros_(module.bias)
    with torch.no_grad():
        router.fm_embed.weight.fill_(10.0)

    fused, stats = router(fm_embs, fm_ids, return_stats=True)

    expected = fm_embs.mean(dim=0)
    assert torch.allclose(stats["fm_weights_local"], torch.full_like(stats["fm_weights_local"], 0.25))
    assert torch.allclose(fused, expected, atol=1e-6)


def test_fm_weight_entropy_is_finite_and_nonnegative():
    router = _construct_router(num_fms=4)
    fm_embs, fm_ids = _create_syn_input(k=3)
    _, stats = router(fm_embs, fm_ids, return_stats=True)
    ent = stats["fm_weight_entropy"]
    assert ent.shape == (2, 3)
    assert torch.isfinite(ent).all()
    assert (ent >= -1e-6).all()


def _build_moco(num_fms: int = 4, **kwargs) -> MoCo:
    return MoCo(
        embed_dim=64,
        contrast_dim=32,
        input_dims=[64],
        num_heads=4,
        num_layers=1,
        T=0.2,
        dropout=0.0,
        sequence_encoder="transformer",
        pooling="abmil",
        fm_pooling="router",
        num_fms=num_fms,
        att_dim=32,
        router_load_balance_weight=1.0,
        router_z_loss_weight=1.0,
        router_entropy_weight=1.0,
        **kwargs,
    ).to(DEVICE)


def _entropy_from_weights(weights: torch.Tensor) -> torch.Tensor:
    probs = weights.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1)


def _reference_load_balance(
    num_fms: int,
    fm_weights_local: torch.Tensor,
    fm_ids: torch.Tensor,
    slice_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    if slice_mask is not None:
        slice_valid = slice_mask.to(dtype=fm_weights_local.dtype)
        denom = slice_valid.sum(dim=1).clamp_min(1.0)
        importance = (fm_weights_local * slice_valid.unsqueeze(-1)).sum(dim=1) / denom[:, None]
    else:
        importance = fm_weights_local.mean(dim=1)

    valid = torch.ones_like(importance)
    per_sample_k = valid.sum(dim=1).clamp_min(1.0)
    target_local = valid / per_sample_k[:, None]

    flat_ids = fm_ids.reshape(-1).long()
    device = fm_weights_local.device
    usage = torch.zeros(num_fms, dtype=importance.dtype, device=device)
    target = torch.zeros(num_fms, dtype=importance.dtype, device=device)
    count = torch.zeros(num_fms, dtype=importance.dtype, device=device)
    usage.index_add_(0, flat_ids, (importance * valid).reshape(-1))
    target.index_add_(0, flat_ids, target_local.reshape(-1))
    count.index_add_(0, flat_ids, valid.reshape(-1))

    active = count > 0
    k_active = int(active.sum().item())
    if k_active <= 1:
        return torch.zeros((), dtype=importance.dtype, device=device), {"k_active": k_active}

    mean_usage = usage[active] / count[active].clamp_min(1e-8)
    mean_target = target[active] / count[active].clamp_min(1e-8)
    sq_err = (mean_usage - mean_target) ** 2
    return k_active * sq_err.sum(), {
        "importance": importance,
        "target_local": target_local,
        "mean_usage": mean_usage,
        "mean_target": mean_target,
        "sq_err": sq_err,
        "k_active": k_active,
    }


def _make_stats(
    weights: torch.Tensor,
    fm_ids: torch.Tensor,
    router_logits: torch.Tensor,
    slice_mask: torch.Tensor | None = None,
) -> dict:
    return {
        "fm_weights_local": weights,
        "fm_ids": fm_ids,
        "router_logits": router_logits,
        "fm_weight_entropy": _entropy_from_weights(weights),
        "slice_mask": slice_mask,
    }


def _print_loss_case(title: str, **fields) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)
    for key, value in fields.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.detach().cpu().tolist()}")
        else:
            print(f"  {key}: {value}")


def test_router_aux_loss_diagnostic_report():
    """
    Print load-balance, z-loss, and entropy on synthetic tensors.
    """
    model = _build_moco(num_fms=4).eval()

    # Case 1: uniform 2-FM weights -> load-balance = 0
    weights_uniform = torch.full((1, 3, 2), 0.5, device=DEVICE)
    fm_ids_2 = torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE)
    logits_zero = torch.zeros(1, 3, 2, device=DEVICE)
    stats_uniform = _make_stats(weights_uniform, fm_ids_2, logits_zero)

    lb_uniform = model._fm_load_balance_loss(weights_uniform, fm_ids_2)
    ref_uniform, dbg_uniform = _reference_load_balance(4, weights_uniform, fm_ids_2)
    z_uniform = MoCo._router_z_loss(logits_zero)
    ent_uniform = stats_uniform["fm_weight_entropy"].mean()

    _print_loss_case(
        "Case 1 — uniform 2-FM weights",
        weights=weights_uniform,
        importance=dbg_uniform["importance"],
        target_local=dbg_uniform["target_local"],
        load_balance=lb_uniform.item(),
        z_loss=z_uniform.item(),
        expected_z=math.log(2) ** 2,
        mean_entropy=ent_uniform.item(),
        expected_entropy=math.log(2),
    )
    assert torch.allclose(lb_uniform, ref_uniform, atol=1e-6)
    assert lb_uniform.item() == pytest.approx(0.0, abs=1e-6)
    assert z_uniform.item() == pytest.approx(math.log(2) ** 2, rel=1e-5)
    assert ent_uniform.item() == pytest.approx(math.log(2), rel=1e-5)

    # Case 2: skewed weights -> LB = 2 * (0.4^2 + 0.4^2) = 0.64
    weights_skew = torch.tensor([[[0.9, 0.1], [0.9, 0.1]]], dtype=torch.float32, device=DEVICE)
    lb_skew = model._fm_load_balance_loss(weights_skew, fm_ids_2)
    ref_skew, dbg_skew = _reference_load_balance(4, weights_skew, fm_ids_2)
    expected_skew = 0.64

    _print_loss_case(
        "Case 2 — skewed 2-FM weights (0.9 / 0.1)",
        mean_usage=dbg_skew["mean_usage"],
        mean_target=dbg_skew["mean_target"],
        sq_err=dbg_skew["sq_err"],
        load_balance=lb_skew.item(),
        expected_load_balance=expected_skew,
    )
    assert lb_skew.item() == pytest.approx(expected_skew, rel=1e-5)

    # Case 3: FMRouter stats feed the same entropy formula
    router = _construct_router(num_fms=4)
    fm_embs, fm_ids = _create_syn_input(k=2, batch=1, num_slices=2)
    _, router_stats = router(fm_embs, fm_ids, return_stats=True)
    router_ent = router_stats["fm_weight_entropy"].mean()
    recomputed_ent = _entropy_from_weights(router_stats["fm_weights_local"]).mean()

    _print_loss_case(
        "Case 3 — FMRouter entropy matches -sum(w log w)",
        router_entropy=router_ent.item(),
        recomputed_entropy=recomputed_ent.item(),
    )
    assert torch.allclose(router_ent, recomputed_ent, atol=1e-5)

    # Case 4: z-loss ignores padded slices
    real_logits = torch.zeros(1, 2, 2, device=DEVICE)
    padded_logits = torch.cat([real_logits, torch.full((1, 2, 2), 50.0, device=DEVICE)], dim=1)
    slice_mask = torch.tensor([[True, True, False, False]], device=DEVICE)
    z_masked = MoCo._router_z_loss(padded_logits, slice_mask)
    z_expected = MoCo._router_z_loss(real_logits)

    _print_loss_case(
        "Case 4 — z-loss = mean(logsumexp(logits, dim=-1)^2), masked slices",
        z_masked=z_masked.item(),
        z_expected=z_expected.item(),
    )
    assert torch.allclose(z_masked, z_expected, atol=1e-6)

    # Case 5: _collect_router_losses averages two contrastive views
    stats_view1 = _make_stats(weights_skew, fm_ids_2, torch.zeros(1, 2, 2, device=DEVICE))
    stats_view2 = _make_stats(weights_uniform, fm_ids_2, torch.zeros(1, 3, 2, device=DEVICE))
    collected = model._collect_router_losses([stats_view1, stats_view2])
    aux = model._router_aux_loss(torch.zeros((), device=DEVICE), [stats_view1, stats_view2])

    _print_loss_case(
        "Case 5 — MoCo _collect_router_losses (two views)",
        loss_router_balance=collected["loss_router_balance"].item(),
        loss_router_z=collected["loss_router_z"].item(),
        loss_router_entropy=collected["loss_router_entropy"].item(),
        total_aux_loss=aux["loss"].item(),
        formula="loss = infonce + w_bal*LB + w_z*z + w_ent*H  (w=1 here)",
    )

    expected_balance = (lb_skew + lb_uniform) / 2
    expected_z = (
        MoCo._router_z_loss(stats_view1["router_logits"])
        + MoCo._router_z_loss(stats_view2["router_logits"])
    ) / 2
    expected_entropy = (
        stats_view1["fm_weight_entropy"].mean() + stats_view2["fm_weight_entropy"].mean()
    ) / 2
    assert collected["loss_router_balance"].item() == pytest.approx(expected_balance.item(), rel=1e-5)
    assert aux["loss"].item() == pytest.approx(
        expected_balance.item() + expected_z.item() + expected_entropy.item(), rel=1e-5
    )

    print(f"\n{'=' * 72}")
    print("Router auxiliary loss diagnostic checks passed.")
    print("=" * 72)


def test_router_aux_loss_formula():
    model = _build_moco(num_fms=3).eval()

    weights = torch.tensor([[[0.25, 0.75]]], device=DEVICE)
    ids = torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE)
    lb = model._fm_load_balance_loss(weights, ids)
    assert lb.item() == pytest.approx(0.25, rel=1e-5)

    logits = torch.tensor([[[0.0, 0.0], [1.0, -1.0]]], device=DEVICE)
    z = MoCo._router_z_loss(logits)
    per_slice = torch.logsumexp(logits.float(), dim=-1) ** 2
    assert z.item() == pytest.approx(per_slice.mean().item(), rel=1e-5)

