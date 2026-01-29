import torch
import pytest
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

from med_slim.model.ssl import MoCo


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomPairDataset(Dataset):
    """
    Tiny synthetic dataset that yields two random feature bags and their
    per-sample original feature dims for variable-dim support.
    """
    def __init__(self, length: int, num_slices: int, max_feature_dim: int, true_feature_dim: int):
        self.length = length
        self.num_slices = num_slices
        self.max_feature_dim = max_feature_dim
        self.true_feature_dim = true_feature_dim

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
        return x1, size1, x2, size2

class RandomPairDatasetVariableLength(Dataset):
    """
    Synthetic dataset with variable-length sequences for testing packed sequences.
    Each sample has a different sequence length.
    """
    def __init__(self, length: int, min_slices: int, max_slices: int, feature_dim: int):
        self.length = length
        self.min_slices = min_slices
        self.max_slices = max_slices
        self.feature_dim = feature_dim
        # Pre-generate random sequence lengths for reproducibility
        torch.manual_seed(42)
        self.seq_lens = torch.randint(min_slices, max_slices + 1, (length,))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq_len = self.seq_lens[idx].item()
        # Variable-length features: [seq_len, feature_dim]
        x1 = torch.randn(seq_len, self.feature_dim, dtype=torch.float32)
        x2 = torch.randn(seq_len, self.feature_dim, dtype=torch.float32)
        return {
            "feats1": x1,
            "feats2": x2,
            "orig_embed_dim1": torch.tensor(self.feature_dim, dtype=torch.long),
            "orig_embed_dim2": torch.tensor(self.feature_dim, dtype=torch.long),
            "seq_len1": torch.tensor(seq_len, dtype=torch.long),
            "seq_len2": torch.tensor(seq_len, dtype=torch.long),
        }

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
    for x1, size1, x2, size2 in loader:
        x1 = x1.to(DEVICE, dtype=torch.float32)
        x2 = x2.to(DEVICE, dtype=torch.float32)
        size1 = size1.to(DEVICE, dtype=torch.long)
        size2 = size2.to(DEVICE, dtype=torch.long)

        loss = model(x1, x2, input_feature_dims_1=size1, input_feature_dims_2=size2, m=0.99)
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


def test_training_packed_sequences():
    """Test training with packed sequences (variable-length without padding)."""
    from med_slim.data import ssl_packed_collate_fn
    
    torch.manual_seed(0)

    # Tiny problem size for speed
    batch_size = 4
    num_batches = 2
    min_slices = 4
    max_slices = 16  # Variable sequence lengths
    feature_dim = 16
    
    # FlashAttention requires bf16 or fp16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    accelerator = Accelerator(cpu=not torch.cuda.is_available())

    # Model with transformer encoder (to test VarlenTransformerEncoder)
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
        d_state=64,
        sequence_encoder="transformer",  # Test with transformer
    ).to(DEVICE, dtype=dtype).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    ds = RandomPairDatasetVariableLength(
        length=batch_size * num_batches,
        min_slices=min_slices,
        max_slices=max_slices,
        feature_dim=feature_dim,
    )
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True,
        collate_fn=ssl_packed_collate_fn,  # Use packed collate
    )

    # Snapshot parameters before training
    params_before = _clone_params(model)

    total_loss = 0.0
    steps = 0
    for batch in loader:
        # Move packed data to device (FlashAttention requires bf16/fp16)
        feats1 = batch["feats1"].to(DEVICE, dtype=dtype)
        feats2 = batch["feats2"].to(DEVICE, dtype=dtype)
        cu_seqlens1 = batch["cu_seqlens1"].to(DEVICE)
        cu_seqlens2 = batch["cu_seqlens2"].to(DEVICE)
        max_seqlen1 = batch["max_seqlen1"]
        max_seqlen2 = batch["max_seqlen2"]
        orig_embed_dim1 = batch["orig_embed_dim1"].to(DEVICE)
        orig_embed_dim2 = batch["orig_embed_dim2"].to(DEVICE)
        seq_idx1 = batch["seq_idx1"].to(DEVICE)
        seq_idx2 = batch["seq_idx2"].to(DEVICE)

        # Use forward_packed
        loss = model.forward_packed(
            feats1, feats2,
            cu_seqlens1, cu_seqlens2,
            max_seqlen1, max_seqlen2,
            input_feature_dims_1=orig_embed_dim1,
            input_feature_dims_2=orig_embed_dim2,
            seq_idx1=seq_idx1,
            seq_idx2=seq_idx2,
            m=0.99,
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
    assert changed, "Model parameters did not update during packed sequence training"


def test_training_packed_sequences_mamba2():
    """Test training with packed sequences using Mamba2 encoder."""
    from med_slim.data import ssl_packed_collate_fn
    
    torch.manual_seed(0)

    batch_size = 4
    num_batches = 2
    min_slices = 4
    max_slices = 16
    feature_dim = 16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    accelerator = Accelerator(cpu=not torch.cuda.is_available())

    # Model with Mamba2 encoder
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
        sequence_encoder="mamba2",  # Test with mamba2
    ).to(DEVICE, dtype=dtype).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    ds = RandomPairDatasetVariableLength(
        length=batch_size * num_batches,
        min_slices=min_slices,
        max_slices=max_slices,
        feature_dim=feature_dim,
    )
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True,
        collate_fn=ssl_packed_collate_fn,
    )

    params_before = _clone_params(model)

    total_loss = 0.0
    steps = 0
    for batch in loader:
        feats1 = batch["feats1"].to(DEVICE, dtype=dtype)
        feats2 = batch["feats2"].to(DEVICE, dtype=dtype)
        cu_seqlens1 = batch["cu_seqlens1"].to(DEVICE)
        cu_seqlens2 = batch["cu_seqlens2"].to(DEVICE)
        max_seqlen1 = batch["max_seqlen1"]
        max_seqlen2 = batch["max_seqlen2"]
        orig_embed_dim1 = batch["orig_embed_dim1"].to(DEVICE)
        orig_embed_dim2 = batch["orig_embed_dim2"].to(DEVICE)
        seq_idx1 = batch["seq_idx1"].to(DEVICE)
        seq_idx2 = batch["seq_idx2"].to(DEVICE)

        loss = model.forward_packed(
            feats1, feats2,
            cu_seqlens1, cu_seqlens2,
            max_seqlen1, max_seqlen2,
            input_feature_dims_1=orig_embed_dim1,
            input_feature_dims_2=orig_embed_dim2,
            seq_idx1=seq_idx1,
            seq_idx2=seq_idx2,
            m=0.99,
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
    assert changed, "Model parameters did not update during Mamba2 packed sequence training"


def test_packed_collate_fn_shapes():
    """Test that ssl_packed_collate_fn produces correct shapes."""
    from med_slim.data import ssl_packed_collate_fn
    
    batch_size = 4
    feature_dim = 16
    
    ds = RandomPairDatasetVariableLength(
        length=batch_size,
        min_slices=4,
        max_slices=16,
        feature_dim=feature_dim,
    )
    
    batch = [ds[i] for i in range(batch_size)]
    collated = ssl_packed_collate_fn(batch)
    
    # Check all expected keys exist
    expected_keys = {
        "feats1", "feats2", 
        "cu_seqlens1", "cu_seqlens2",
        "max_seqlen1", "max_seqlen2",
        "seq_idx1", "seq_idx2",
        "orig_embed_dim1", "orig_embed_dim2",
        "batch_size",
    }
    assert set(collated.keys()) == expected_keys, f"Missing keys: {expected_keys - set(collated.keys())}"
    
    # Check shapes
    total_tokens1 = collated["cu_seqlens1"][-1].item()
    total_tokens2 = collated["cu_seqlens2"][-1].item()
    
    assert collated["feats1"].shape == (total_tokens1, feature_dim)
    assert collated["feats2"].shape == (total_tokens2, feature_dim)
    assert collated["cu_seqlens1"].shape == (batch_size + 1,)
    assert collated["cu_seqlens2"].shape == (batch_size + 1,)
    assert collated["seq_idx1"].shape == (total_tokens1,)
    assert collated["seq_idx2"].shape == (total_tokens2,)
    assert collated["orig_embed_dim1"].shape == (batch_size,)
    assert collated["orig_embed_dim2"].shape == (batch_size,)
    assert collated["batch_size"] == batch_size
    
    # Verify cu_seqlens is monotonically increasing starting from 0
    assert collated["cu_seqlens1"][0] == 0
    assert collated["cu_seqlens2"][0] == 0
    assert (collated["cu_seqlens1"][1:] > collated["cu_seqlens1"][:-1]).all()
    assert (collated["cu_seqlens2"][1:] > collated["cu_seqlens2"][:-1]).all()


