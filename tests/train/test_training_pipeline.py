import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch._dynamo
import torch._inductor.config as inductor_config
import pytest
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

from med_slim.model.ssl import MoCo
from med_slim.data import ssl_packed_collate_fn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomPairDataset(Dataset):
    """
    Tiny synthetic dataset that yields two random feature bags and their
    per-sample original feature dims for variable-dim support.
    Optionally includes labels for semi-supervised contrastive learning testing.
    """
    def __init__(self, length: int, num_slices: int, max_feature_dim: int, true_feature_dim: int,
                 num_classes: int = 0, label_ratio: float = 0.5):
        self.length = length
        self.num_slices = num_slices
        self.max_feature_dim = max_feature_dim
        self.true_feature_dim = true_feature_dim
        self.num_classes = num_classes
        self.label_ratio = label_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Create padded features with last-dim = max_feature_dim but only the first
        # true_feature_dim columns contain signal; the rest are zeros.
        x1 = torch.zeros(self.num_slices, self.max_feature_dim, dtype=torch.float32)
        x2 = torch.zeros(self.num_slices, self.max_feature_dim, dtype=torch.float32)
        x1[:, : self.true_feature_dim] = torch.randn(self.num_slices, self.true_feature_dim, dtype=torch.float32)
        x2[:, : self.true_feature_dim] = torch.randn(self.num_slices, self.true_feature_dim, dtype=torch.float32)
        size1 = torch.tensor(self.true_feature_dim, dtype=torch.long)
        size2 = torch.tensor(self.true_feature_dim, dtype=torch.long)
        result = {
            "feats1": x1,
            "feats2": x2,
            "orig_embed_dim1": size1,
            "orig_embed_dim2": size2,
            "seq_len": torch.tensor(self.num_slices, dtype=torch.long),
        }
        
        if self.num_classes > 0:
            # Deterministic labeling based on index for reproducibility
            is_labeled = (idx % int(1 / self.label_ratio) == 0) if self.label_ratio > 0 else False
            if is_labeled:
                # Generate a deterministic label pattern based on idx
                label = torch.zeros(self.num_classes, dtype=torch.float32)
                label[idx % self.num_classes] = 1.0
                result["label"] = label
                result["has_label"] = torch.tensor(True)
            else:
                result["label"] = torch.zeros(self.num_classes, dtype=torch.float32)
                result["has_label"] = torch.tensor(False)
        
        return result

def _clone_params(model: torch.nn.Module):
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def test_training_one_epoch_random_data():
    torch.manual_seed(0)

    # Tiny problem size for speed
    batch_size = 2
    num_batches = 2  # total steps = 4
    num_slices = 8
    feature_dim = 16  # true feature dimension
    max_feature_dim = feature_dim  # keep equal for simplicity

    # Create accelerator for MoCo
    accelerator = Accelerator(cpu=not torch.cuda.is_available())

    # Model kept intentionally small/lightweight
    model = MoCo(
        embed_dim=256,
        contrast_dim=16,
        accelerator=accelerator,
        input_dims=[feature_dim],
        num_heads=2,
        num_layers=1,
        T=0.2,
        dropout=0.0,
        att_dim=32,
        d_state=64,
    ).to(DEVICE).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    ds = RandomPairDataset(length=batch_size * num_batches, num_slices=num_slices,
                           max_feature_dim=max_feature_dim, true_feature_dim=feature_dim)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # Snapshot parameters before training to verify they update
    params_before = _clone_params(model)

    total_loss = 0.0
    steps = 0
    for batch in loader:
        x1 = batch["feats1"].to(DEVICE, dtype=torch.float32)
        x2 = batch["feats2"].to(DEVICE, dtype=torch.float32)
        size1 = batch["orig_embed_dim1"].to(DEVICE, dtype=torch.long)
        size2 = batch["orig_embed_dim2"].to(DEVICE, dtype=torch.long)
        seq_lens = batch["seq_len"].to(DEVICE, dtype=torch.long)

        loss = model(
            x1, x2,
            input_feature_dims_1=size1, input_feature_dims_2=size2,
            seq_lengths=seq_lens,
            m=0.99,
        )
        assert torch.isfinite(loss).all()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        steps += 1

    assert steps == num_batches
    avg_loss = total_loss / steps
    assert avg_loss > 0.0 and torch.isfinite(torch.tensor(avg_loss))

    # Verify parameters have changed after one epoch
    params_after = [p.detach() for p in model.parameters() if p.requires_grad]
    changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
    assert changed, "Model parameters did not update during training"


def test_training_with_transformer_encoder():
    """Test training with transformer encoder and fixed-length sequences."""
    torch.manual_seed(0)

    batch_size = 4
    num_batches = 2
    num_slices = 8
    feature_dim = 16
    
    accelerator = Accelerator(cpu=not torch.cuda.is_available())

    model = MoCo(
        embed_dim=64,
        contrast_dim=16,
        accelerator=accelerator,
        input_dims=[feature_dim],
        num_heads=2,
        num_layers=1,
        T=0.2,
        dropout=0.0,
        att_dim=32,
        sequence_encoder="transformer",
    ).to(DEVICE).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    ds = RandomPairDataset(
        length=batch_size * num_batches,
        num_slices=num_slices,
        max_feature_dim=feature_dim,
        true_feature_dim=feature_dim,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    params_before = _clone_params(model)

    total_loss = 0.0
    steps = 0
    for batch in loader:
        x1 = batch["feats1"].to(DEVICE, dtype=torch.float32)
        x2 = batch["feats2"].to(DEVICE, dtype=torch.float32)
        size1 = batch["orig_embed_dim1"].to(DEVICE, dtype=torch.long)
        size2 = batch["orig_embed_dim2"].to(DEVICE, dtype=torch.long)
        seq_lens = batch["seq_len"].to(DEVICE, dtype=torch.long)

        loss = model(
            x1, x2,
            input_feature_dims_1=size1, input_feature_dims_2=size2,
            seq_lengths=seq_lens,
            m=0.99,
            use_packed=True,
            cu_seqlens1=cu_seqlens1, cu_seqlens2=cu_seqlens2,
            max_seqlen1=max_seqlen1, max_seqlen2=max_seqlen2,
            seq_idx1=seq_idx1, seq_idx2=seq_idx2,
        )
        assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        steps += 1

    assert steps == num_batches
    avg_loss = total_loss / steps
    assert avg_loss > 0.0 and torch.isfinite(torch.tensor(avg_loss))

    params_after = [p.detach() for p in model.parameters() if p.requires_grad]
    changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
    assert changed, "Model parameters did not update during transformer training"


def test_training_semi_supervised_with_labels():
    """Test training with semi-supervised contrastive learning (mixed labeled/unlabeled)."""
    torch.manual_seed(0)

    batch_size = 4
    num_batches = 2
    num_slices = 8
    feature_dim = 16
    num_classes = 3 
    
    accelerator = Accelerator(cpu=not torch.cuda.is_available())

    model = MoCo(
        embed_dim=256,
        contrast_dim=16,
        accelerator=accelerator,
        input_dims=[feature_dim],
        num_heads=2,
        num_layers=1,
        T=0.2,
        dropout=0.0,
        att_dim=32,
        d_state=64,
    ).to(DEVICE).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    # 50% of samples have labels
    ds = RandomPairDataset(
        length=batch_size * num_batches,
        num_slices=num_slices,
        max_feature_dim=feature_dim,
        true_feature_dim=feature_dim,
        num_classes=num_classes,
        label_ratio=0.5,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    params_before = _clone_params(model)

    total_loss = 0.0
    steps = 0
    for batch in loader:
        x1 = batch["feats1"].to(DEVICE, dtype=torch.float32)
        x2 = batch["feats2"].to(DEVICE, dtype=torch.float32)
        size1 = batch["orig_embed_dim1"].to(DEVICE, dtype=torch.long)
        size2 = batch["orig_embed_dim2"].to(DEVICE, dtype=torch.long)
        seq_lens = batch["seq_len"].to(DEVICE, dtype=torch.long)
        labels = batch["label"].to(DEVICE, dtype=torch.float32)
        has_label = batch["has_label"].to(DEVICE)

        loss = model(
            x1, x2,
            input_feature_dims_1=size1, input_feature_dims_2=size2,
            seq_lengths=seq_lens,
            m=0.99,
            labels=labels,
            has_label=has_label,
        )
        assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        steps += 1

    assert steps == num_batches
    avg_loss = total_loss / steps
    assert avg_loss > 0.0 and torch.isfinite(torch.tensor(avg_loss))

    # Verify parameters have changed
    params_after = [p.detach() for p in model.parameters() if p.requires_grad]
    changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
    assert changed, "Model parameters did not update during semi-supervised training"


def test_training_semi_supervised_all_labeled():
    """Test training when all samples in the batch are labeled (pure SupCon)."""
    torch.manual_seed(0)

    batch_size = 4
    num_batches = 2
    num_slices = 8
    feature_dim = 16
    num_classes = 3

    accelerator = Accelerator(cpu=not torch.cuda.is_available())

    model = MoCo(
        embed_dim=256,
        contrast_dim=16,
        accelerator=accelerator,
        input_dims=[feature_dim],
        num_heads=2,
        num_layers=1,
        T=0.2,
        dropout=0.0,
        att_dim=32,
        d_state=64,
    ).to(DEVICE).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    ds = RandomPairDataset(
        length=batch_size * num_batches,
        num_slices=num_slices,
        max_feature_dim=feature_dim,
        true_feature_dim=feature_dim,
        num_classes=num_classes,
        label_ratio=1.0,  # All labeled
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    total_loss = 0.0
    steps = 0
    for batch in loader:
        x1 = batch["feats1"].to(DEVICE, dtype=torch.float32)
        x2 = batch["feats2"].to(DEVICE, dtype=torch.float32)
        labels = batch["label"].to(DEVICE, dtype=torch.float32)
        has_label = batch["has_label"].to(DEVICE)

        loss = model(x1, x2, m=0.99, labels=labels, has_label=has_label)
        assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        steps += 1

    assert steps == num_batches
    avg_loss = total_loss / steps
    assert avg_loss > 0.0 and torch.isfinite(torch.tensor(avg_loss))
