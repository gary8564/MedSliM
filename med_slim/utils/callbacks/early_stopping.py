import torch
from typing import Optional
from accelerate import Accelerator
import logging
from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stops the training if monitored metric doesn't improve after a given patience.
    """
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
        ckpt_path: str = "best.pt",
        accelerator: Optional[Accelerator] = None
    ):
        """
        Args:
            patience: how many epochs to wait after last time metric improved.
            min_delta: minimum change in the monitored metric to qualify as improvement.
            mode: one of {"min", "max"}. In 'min' mode, lower metric is better.
            ckpt_path: where to save the best model checkpoint.
        """
        assert mode in ("min", "max"), f"`mode` attribute must be specified either `min` or `max`."
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.ckpt_path = ckpt_path
        self.accelerator = accelerator

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(
        self,
        val_score: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        epoch: int
    ):
        score = val_score if self.mode == "max" else -val_score

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, optimizer, scheduler, epoch, score)
            return self.early_stop, self.best_score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop, self.best_score
        else:
            self.best_score = score
            self._save_checkpoint(model, optimizer, scheduler, epoch, score)
            self.counter = 0
            return self.early_stop, self.best_score

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        epoch: int,
        best_metric: float
    ):
        """Saves model when metric improves."""
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        try:
            # Get the unwrapped model state
            if self.accelerator is not None:
                model_state = self.accelerator.get_state_dict(model)
            else:
                model_state = model.state_dict()
            # Save checkpoint directly
            checkpoint_data = {
                "epoch": epoch,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "best_metric": best_metric,
            }
            torch.save(checkpoint_data, self.ckpt_path)
            logger.info(f"New best validation metric = {best_metric:.4f} at epoch {epoch+1}, saved best.pt")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_best_model(self, model: torch.nn.Module):
        """Loads the best saved model weights into `model`."""
        try:
            # Load checkpoint with proper device placement
            ckpt = torch.load(self.ckpt_path, map_location='cpu')
            # Load state dict to model
            model.load_state_dict(ckpt["model_state"])
            return model
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise