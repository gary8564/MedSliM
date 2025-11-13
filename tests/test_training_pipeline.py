import torch
import pytest
from torch.utils.data import Dataset, DataLoader

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

def _clone_params(model: torch.nn.Module):
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


@pytest.mark.timeout(30)
def test_training_one_epoch_random_data():
    torch.manual_seed(0)

    # Tiny problem size for speed
    batch_size = 2
    num_batches = 2  # total steps = 4
    num_slices = 8
    feature_dim = 16  # true feature dimension
    max_feature_dim = feature_dim  # keep equal for simplicity

    # Model kept intentionally small/lightweight
    model = MoCo(
        embed_dim=256,
        contrast_dim=16,
        input_dims=[feature_dim],
        num_heads=2,
        num_mamba_layers=1,
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


