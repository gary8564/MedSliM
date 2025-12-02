import torch
import pytest
from unittest.mock import MagicMock, patch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

from med_slim.model.sequence_encoder.cobra import Cobra
from med_slim.eval.linear_probing import (
    LinearClassifier,
    train_linear_classifier,
    evaluate_classifier,
    _compute_loss,
    _compute_metrics,
    train_per_epoch,
    eval_per_epoch,
)
from med_slim.eval.extract_feats import get_cobra_feats
from med_slim.utils.metrics.linear import get_loss_criterion


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyFeatDataset(Dataset):
    """
    Dummy dataset mimicking FeatClassificationDataset output.
    Returns features from multiple encoders for each sample.
    """
    def __init__(self, n_samples: int, n_encoders: int, num_slices: int, 
                 embed_dims: list, task: str = "binary", num_classes: int = 2):
        self.n_samples = n_samples
        self.n_encoders = n_encoders
        self.num_slices = num_slices
        self.embed_dims = embed_dims  # List of embed dims per encoder
        self.task = task
        self.num_classes = num_classes
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Generate features for each encoder with their respective embed_dim
        feat_embeds = [
            torch.randn(self.num_slices, self.embed_dims[i])
            for i in range(self.n_encoders)
        ]
        
        # Generate label based on task
        if self.task == "multilabel":
            label = torch.randint(0, 2, (self.num_classes,)).float()
        else:
            label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        
        return {"feature_embeds": feat_embeds, "label": label, "sample_id": idx}


def dummy_collate_fn(batch):
    """Collate function for DummyFeatDataset."""
    labels = torch.stack([item["label"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]
    
    n_encoders = len(batch[0]["feature_embeds"])
    collated_feats = []
    for k in range(n_encoders):
        kth_feats = torch.stack([item["feature_embeds"][k] for item in batch])
        collated_feats.append(kth_feats)
    
    return {"feature_embeds": collated_feats, "labels": labels, "sample_ids": sample_ids}


# -------------------- Unit Tests --------------------
def test_linear_classifier_binary():
    clf = LinearClassifier(input_dim=64, num_classes=2, hidden_dim=32).to(DEVICE)
    x = torch.randn(8, 64, device=DEVICE)
    out = clf(x)
    assert out.shape == (8, 1)


def test_linear_classifier_multiclass():
    clf = LinearClassifier(input_dim=64, num_classes=5, hidden_dim=32).to(DEVICE)
    x = torch.randn(8, 64, device=DEVICE)
    out = clf(x)
    assert out.shape == (8, 5)


def test_compute_loss_binary():
    logits = torch.randn(8, 1, device=DEVICE)
    labels = torch.randint(0, 2, (8,), device=DEVICE).float()
    criterion = get_loss_criterion("binary")
    loss = _compute_loss(logits, labels, "binary", criterion)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_compute_metrics_binary():
    logits = torch.randn(16, 1, device=DEVICE)
    labels = torch.randint(0, 2, (16,), device=DEVICE).long()
    metrics = _compute_metrics(logits, labels, "binary", 2, DEVICE)
    assert "auroc" in metrics


@pytest.fixture
def mock_wandb():
    """Mock wandb to avoid requiring initialization in tests."""
    with patch("med_slim.eval.linear_probing.wandb") as mock:
        mock.run = MagicMock()
        mock.log = MagicMock()
        yield mock


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Clear GPU memory cache between tests to prevent OOM errors."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------- End-to-End Pipeline Test --------------------
def test_linear_probing_pipeline_binary(mock_wandb):
    """
    End-to-end test: COBRA feature extraction -> linear classifier training -> evaluation.
    """
    torch.manual_seed(42)
    
    # Config
    n_samples = 20
    n_encoders = 2
    num_slices = 8
    embed_dims = [512, 768]  # Different dims per encoder
    batch_size = 4
    cobra_embed_dim = 256
    
    # Create dummy dataset
    train_ds = DummyFeatDataset(n_samples, n_encoders, num_slices, embed_dims, task="binary")
    test_ds = DummyFeatDataset(n_samples // 2, n_encoders, num_slices, embed_dims, task="binary")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=dummy_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=dummy_collate_fn)
    
    # Setup accelerator first to get the device
    accelerator = Accelerator()
    
    # Create COBRA model (inference mode, randomly initialized)
    cobra_model = Cobra(
        embed_dim=cobra_embed_dim,
        input_dims=embed_dims,
        num_heads=2,
        layer=1,
        mode="inference"
    ).to(accelerator.device).eval()
    
    for param in cobra_model.parameters():
        param.requires_grad = False
    
    # Extract COBRA features
    train_embeddings, train_labels, train_ids = get_cobra_feats(cobra_model, train_loader, accelerator)
    test_embeddings, test_labels, test_ids = get_cobra_feats(cobra_model, test_loader, accelerator)
    
    assert train_embeddings.shape == (n_samples, cobra_embed_dim)
    assert test_embeddings.shape == (n_samples // 2, cobra_embed_dim)
    
    # Create config for training
    cfg = {
        "linear_probing": {
            "task": "binary",
            "target_columns": ["label"],
            "hyperparams": {
                "batch_size": 4,
                "hidden_dim": 64,
                "dropout": 0.1,
                "lr": 1e-3,
                "warmup_ratio": 0.1,
                "max_epochs": 2,
                "patience": 5,
            }
        }
    }
    
    # Split train into train/val
    n_val = n_samples // 4
    val_embeddings = train_embeddings[:n_val]
    val_labels = train_labels[:n_val]
    train_emb = train_embeddings[n_val:]
    train_lbl = train_labels[n_val:]
    
    # Train classifier
    classifier = LinearClassifier(
        input_dim=cobra_embed_dim,
        num_classes=2,
        hidden_dim=64,
        dropout=0.1
    )
    
    # Use temporary directory for output
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        training_results = train_linear_classifier(
            classifier,
            train_emb,
            train_lbl,
            val_embeddings,
            val_labels,
            cfg,
            accelerator,
            output_dir=tmpdir,
            fold=0
        )
    
    assert "best_model" in training_results
    assert "best_val_auroc" in training_results
    best_classifier = training_results["best_model"]
    
    # Evaluate on test set
    # For testing, use a temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_results = evaluate_classifier(
            best_classifier,
            test_embeddings,
            test_labels,
            test_ids,
            cfg,
            accelerator,
            output_dir=tmpdir
        )
    
    assert "metrics" in eval_results
    assert "predictions" in eval_results
    assert "auroc" in eval_results["metrics"]
    assert "auprc" in eval_results["metrics"]
    assert len(eval_results["predictions"]) == n_samples // 2


def test_linear_probing_pipeline_multilabel(mock_wandb):
    """
    End-to-end test for multilabel classification.
    """
    torch.manual_seed(42)
    
    n_samples = 16
    n_encoders = 2
    num_slices = 8
    embed_dims = [512, 768]
    batch_size = 4
    cobra_embed_dim = 256
    num_classes = 3
    
    # Create dummy dataset
    train_ds = DummyFeatDataset(n_samples, n_encoders, num_slices, embed_dims, 
                                 task="multilabel", num_classes=num_classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=dummy_collate_fn)
    
    # Setup accelerator first to get the device
    accelerator = Accelerator()
    
    # Create COBRA model
    cobra_model = Cobra(
        embed_dim=cobra_embed_dim,
        input_dims=embed_dims,
        num_heads=2,
        layer=1,
        mode="inference"
    ).to(accelerator.device).eval()
    
    for param in cobra_model.parameters():
        param.requires_grad = False
    
    # Extract features
    embeddings, labels, _ = get_cobra_feats(cobra_model, train_loader, accelerator)
    
    assert embeddings.shape == (n_samples, cobra_embed_dim)
    assert labels.shape == (n_samples, num_classes)
    
    # Quick training test
    cfg = {
        "linear_probing": {
            "task": "multilabel",
            "target_columns": ["label1", "label2", "label3"],
            "hyperparams": {
                "batch_size": 4,
                "hidden_dim": 64,
                "dropout": 0.1,
                "lr": 1e-3,
                "warmup_ratio": 0.1,
                "max_epochs": 1,
                "patience": 5,
            }
        }
    }
    
    n_val = 4
    classifier = LinearClassifier(input_dim=cobra_embed_dim, num_classes=num_classes, hidden_dim=64)
    
    # Use temporary directory for output
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        training_results = train_linear_classifier(
            classifier,
            embeddings[n_val:],
            labels[n_val:],
            embeddings[:n_val],
            labels[:n_val],
            cfg,
            accelerator,
            output_dir=tmpdir,
            fold=0
        )
    
    assert "best_model" in training_results
    assert training_results["best_model"] is not None
