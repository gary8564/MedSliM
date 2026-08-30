"""
Unit tests for the xMIL-style slice-drop explanations.

All tests run on a tiny stub classifier whose logits are a hand-picked linear
function of the kept slices, so the expected deltas and MoRF orders are known in
closed form. No GPU, real COBRA checkpoint, or feature cache is required.
"""

import numpy as np
import pytest
import torch

from med_slim.eval.xai.drop import (
    BagInputs,
    bag_from_batch,
    drop_slices,
    predict_class_scores,
    score_keep,
)
from med_slim.eval.xai.faithfulness import aupc, mean_curve, morf_curve, morf_order
from med_slim.eval.xai.occlusion import (
    leave_one_slice_out,
    roi_group_occlusion,
    sample_control_windows,
)


SEQ_LEN = 6


class SumOfFirstChannelClassifier(torch.nn.Module):
    """
    Stub whose class-1 logit is the sum of the first channel over the kept slices.

    Each slice therefore contributes exactly its own first-channel value, which makes
    every leave-one-out delta equal to that value.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, features, seq_lengths, physical_positions=None):
        seq_len = int(seq_lengths[0].item())
        total = features[0][0, :seq_len, 0].sum()
        logits = torch.zeros(1, self.num_classes)
        logits[0, 1] = total
        return {"logits": logits, "embedding": torch.zeros(1, 1)}


class SingleLogitClassifier(torch.nn.Module):
    """Binary stub emitting one logit, like ClassifierHead with num_classes=2."""

    def forward(self, features, seq_lengths, physical_positions=None):
        seq_len = int(seq_lengths[0].item())
        total = features[0][0, :seq_len, 0].sum()
        return {"logits": total.reshape(1, 1), "embedding": torch.zeros(1, 1)}


def _testcase_bag(slice_values=None, num_fms: int = 2, with_positions: bool = True) -> BagInputs:
    """Bag of K FM tensors whose first channel encodes a known per-slice value."""
    values = torch.arange(1.0, SEQ_LEN + 1.0) if slice_values is None else torch.as_tensor(
        slice_values, dtype=torch.float32
    )
    seq_len = values.shape[0]
    features = []
    for k in range(num_fms):
        # Distinct embed dim per FM so a misaligned gather would break the shapes.
        feat = torch.zeros(1, seq_len, 4 + k)
        feat[0, :, 0] = values
        feat[0, :, 1] = torch.arange(seq_len, dtype=torch.float32)  # slice fingerprint
        features.append(feat)
    positions = (
        torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len) / seq_len
        if with_positions else None
    )
    return BagInputs(
        features=features,
        seq_lengths=torch.tensor([seq_len], dtype=torch.long),
        physical_positions=positions,
    )


# drop_slices
def test_drop_slices_gathers_every_fm_and_positions_consistently():
    bag = _testcase_bag()
    keep = [0, 2, 5]

    dropped = drop_slices(bag.features, bag.seq_lengths, bag.physical_positions, keep)

    assert dropped.is_empty is False
    assert int(dropped.seq_lengths[0]) == len(keep)
    for k, feat in enumerate(dropped.features):
        assert feat.shape == (1, len(keep), 4 + k)
        # The slice fingerprint channel proves each FM kept the same slices.
        assert feat[0, :, 1].tolist() == pytest.approx(keep)
    assert dropped.physical_positions.shape == (1, len(keep))
    assert dropped.physical_positions[0].tolist() == pytest.approx(
        [i / SEQ_LEN for i in keep]
    )


def test_drop_slices_keeps_anatomical_order_and_deduplicates():
    bag = _testcase_bag()
    dropped = drop_slices(bag.features, bag.seq_lengths, bag.physical_positions, [4, 1, 4, 0])
    assert dropped.features[0][0, :, 1].tolist() == pytest.approx([0, 1, 4])


def test_drop_slices_without_physical_positions_returns_none():
    bag = _testcase_bag(with_positions=False)
    dropped = drop_slices(bag.features, bag.seq_lengths, None, [1, 2])
    assert dropped.physical_positions is None
    assert dropped.features[0].shape[1] == 2


def test_drop_slices_empty_keep_set_flags_a_one_token_zero_bag():
    bag = _testcase_bag()
    dropped = drop_slices(bag.features, bag.seq_lengths, bag.physical_positions, [])

    assert dropped.is_empty is True
    assert int(dropped.seq_lengths[0]) == 1
    for feat in dropped.features:
        assert feat.shape[1] == 1
        assert torch.count_nonzero(feat) == 0
    assert dropped.physical_positions.shape == (1, 1)


def test_drop_slices_rejects_out_of_range_index():
    bag = _testcase_bag()
    with pytest.raises(IndexError, match="out of range"):
        drop_slices(bag.features, bag.seq_lengths, bag.physical_positions, [0, SEQ_LEN])


def test_bag_from_batch_strips_padding():
    """Collated padding must not survive into the drop index space."""
    padded = torch.zeros(1, SEQ_LEN + 3, 4)
    padded[0, :SEQ_LEN, 0] = torch.arange(1.0, SEQ_LEN + 1.0)
    batch = {
        "features": [padded],
        "seq_lengths": torch.tensor([SEQ_LEN]),
        "physical_positions": torch.zeros(1, SEQ_LEN + 3),
    }
    bag = bag_from_batch(batch, torch.device("cpu"))
    assert bag.seq_len == SEQ_LEN
    assert bag.features[0].shape[1] == SEQ_LEN
    assert bag.physical_positions.shape[1] == SEQ_LEN


def test_bag_from_batch_rejects_multi_sample_batches():
    batch = {
        "features": [torch.zeros(2, SEQ_LEN, 4)],
        "seq_lengths": torch.tensor([SEQ_LEN, SEQ_LEN]),
        "physical_positions": torch.zeros(2, SEQ_LEN),
    }
    with pytest.raises(ValueError, match="batch_size=1"):
        bag_from_batch(batch, torch.device("cpu"))


# predict_class_scores
def test_predict_class_scores_uses_softmax_of_the_explained_class():
    bag = _testcase_bag()
    model = SumOfFirstChannelClassifier()
    scores = predict_class_scores(
        model, bag.features, bag.seq_lengths, bag.physical_positions, class_index=1
    )
    total = float(sum(range(1, SEQ_LEN + 1)))
    expected = float(torch.softmax(torch.tensor([0.0, total, 0.0]), dim=-1)[1])
    assert scores["logit"] == pytest.approx(total)
    assert scores["prob"] == pytest.approx(expected)


def test_predict_class_scores_flips_the_single_binary_logit_for_class_zero():
    bag = _testcase_bag(slice_values=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    model = SingleLogitClassifier()
    positive = predict_class_scores(model, bag.features, bag.seq_lengths, None, class_index=1)
    negative = predict_class_scores(model, bag.features, bag.seq_lengths, None, class_index=0)
    assert positive["logit"] == pytest.approx(2.0)
    assert negative["logit"] == pytest.approx(-2.0)
    assert positive["prob"] + negative["prob"] == pytest.approx(1.0)


# leave-one-slice-out deltas
def test_leave_one_slice_out_delta_sign_matches_slice_contribution():
    values = [1.0, -2.0, 3.0, 0.0, 0.5, -0.5]
    bag = _testcase_bag(slice_values=values)
    model = SumOfFirstChannelClassifier()

    result = leave_one_slice_out(model, bag, class_index=1)

    # delta = F_full - F_without_t, and this stub's logit is additive in the slices.
    assert result["delta_logit"].tolist() == pytest.approx(values)
    assert result["full"]["logit"] == pytest.approx(sum(values))


def test_score_keep_reports_the_empty_bag_flag():
    bag = _testcase_bag()
    model = SumOfFirstChannelClassifier()
    scores = score_keep(model, bag, [], class_index=1)
    assert scores["is_empty"] is True
    assert scores["num_kept"] == 0
    assert scores["logit"] == pytest.approx(0.0)


# ROI-group occlusion
def test_roi_group_occlusion_isolates_the_roi_contribution():
    values = [0.0, 0.0, 5.0, 5.0, 0.0, 0.0]
    bag = _testcase_bag(slice_values=values)
    model = SumOfFirstChannelClassifier()
    roi_range = {"roi_z": 2, "roi_depth": 2, "roi_start_slice": 2, "roi_end_slice": 3}

    row = roi_group_occlusion(
        model, bag, roi_range, class_index=1, rng=np.random.default_rng(0), n_random=3
    )

    assert row["delta_logit_roi"] == pytest.approx(10.0)
    # Control windows avoid the ROI, so removing them changes nothing here.
    assert row["delta_logit_random"] == pytest.approx(0.0)
    assert row["control_windows_disjoint"] is True
    # Keeping only the ROI loses nothing, since the other slices contribute zero.
    assert row["delta_logit_complement"] == pytest.approx(0.0)
    assert row["roi_gain_over_random"] == pytest.approx(10.0)


def test_sample_control_windows_prefers_disjoint_windows():
    windows = sample_control_windows(
        seq_len=10, roi_start=4, roi_end=5, n_windows=4, rng=np.random.default_rng(1)
    )
    assert len(windows) == 4
    for window in windows:
        assert len(window) == 2
        assert max(window) < 4 or min(window) > 5


def test_sample_control_windows_falls_back_when_no_disjoint_window_fits():
    """A ROI wider than half the bag leaves only overlapping controls."""
    windows = sample_control_windows(
        seq_len=6, roi_start=1, roi_end=4, n_windows=2, rng=np.random.default_rng(2)
    )
    assert len(windows) == 2
    for window in windows:
        assert len(window) == 4
        assert window[0] != 1


def test_sample_control_windows_empty_when_roi_covers_the_bag():
    windows = sample_control_windows(
        seq_len=4, roi_start=0, roi_end=3, n_windows=3, rng=np.random.default_rng(3)
    )
    assert windows == []


# MoRF ordering and curves
def test_morf_order_is_descending_with_index_tiebreak():
    order = morf_order([0.1, 0.9, 0.5, 0.9])
    assert order.tolist() == [1, 3, 2, 0]


def test_morf_order_of_a_flat_heatmap_is_the_identity():
    order = morf_order([0.25] * 4)
    assert order.tolist() == [0, 1, 2, 3]


def test_morf_drop_removes_the_ranked_slices_cumulatively():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    bag = _testcase_bag(slice_values=values)
    model = SumOfFirstChannelClassifier()

    curve = morf_curve(model, bag, values, class_index=1, mode="drop")

    assert curve["order"].tolist() == [5, 4, 3, 2, 1, 0]
    # Step k has dropped the k largest slices, so the logit is the remaining sum.
    remaining = [sum(sorted(values)[: SEQ_LEN - k]) for k in range(SEQ_LEN + 1)]
    assert curve["logits"].tolist() == pytest.approx(remaining)
    assert curve["fractions"].tolist() == pytest.approx(
        [k / SEQ_LEN for k in range(SEQ_LEN + 1)]
    )


def test_morf_add_keeps_only_the_ranked_slices_cumulatively():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    bag = _testcase_bag(slice_values=values)
    model = SumOfFirstChannelClassifier()

    curve = morf_curve(model, bag, values, class_index=1, mode="add")

    added = [sum(sorted(values, reverse=True)[:k]) for k in range(SEQ_LEN + 1)]
    assert curve["logits"].tolist() == pytest.approx(added)
    assert curve["probs"][0] < curve["probs"][-1]


def test_morf_curve_rejects_a_score_vector_of_the_wrong_length():
    bag = _testcase_bag()
    model = SumOfFirstChannelClassifier()
    with pytest.raises(ValueError, match="match the bag length"):
        morf_curve(model, bag, [1.0, 2.0], class_index=1, mode="drop")


def test_morf_curve_rejects_an_unknown_mode():
    bag = _testcase_bag()
    model = SumOfFirstChannelClassifier()
    with pytest.raises(ValueError, match="mode must be one of"):
        morf_curve(model, bag, [0.0] * SEQ_LEN, class_index=1, mode="occlude")


def test_a_faithful_ranking_has_lower_drop_aupc_than_a_bad_one():
    values = [0.0, 0.0, 6.0, 6.0, 0.0, 0.0]
    bag = _testcase_bag(slice_values=values)
    model = SumOfFirstChannelClassifier()

    faithful = morf_curve(model, bag, values, class_index=1, mode="drop")
    inverted = morf_curve(model, bag, [-v for v in values], class_index=1, mode="drop")

    assert aupc(faithful["fractions"], faithful["probs"]) < aupc(
        inverted["fractions"], inverted["probs"]
    )


def test_mean_curve_interpolates_bags_of_different_lengths():
    curves = [
        {"fractions": np.array([0.0, 0.5, 1.0]), "probs": np.array([1.0, 0.5, 0.0])},
        {"fractions": np.array([0.0, 0.25, 0.75, 1.0]), "probs": np.array([1.0, 1.0, 0.0, 0.0])},
    ]
    averaged = mean_curve(curves, num_points=5)
    assert averaged["fractions"].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert averaged["probs"][0] == pytest.approx(1.0)
    assert averaged["probs"][-1] == pytest.approx(0.0)
    assert averaged["probs"].shape == (5,)
