"""
Adapted from:
[1] https://github.com/KatherLab/COBRA/blob/main/cobra/utils/mamba2.py
Lenz, Tim, Peter Neidlinger, Marta Ligero, Georg Wölflein, Marko van Treeck and Jakob Nikolas Kather. 
Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning.
2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 30807-30817, 2024.
[2] https://github.com/facebookresearch/moco-v3
Chen, Xinlei, Saining Xie and Kaiming He.
An Empirical Study of Training Self-Supervised Vision Transformers.
2021 IEEE/CVF International Conference on Computer Vision (ICCV) (2021): 9620-9629.
"""
import torch
from tqdm import tqdm
from pathlib import Path
import os
import builtins
from torch.utils.data import DataLoader
import argparse
import yaml
from jinja2 import Environment, FileSystemLoader
from pprint import pprint
from datetime import datetime
import math
from accelerate import Accelerator
import wandb

from med_slim.model.ssl import MoCo
from med_slim.data import PrecomputedFeatPairDataset, ssl_collate_fn

CURR_TIME = datetime.now().strftime("%Y-%m-%d-%H:%M")

def main(args, cfg):

    accelerator = Accelerator()
    device = accelerator.device

    if not accelerator.is_main_process:
        def print_pass(*_args, **_kwargs):
            pass
        builtins.print = print_pass

    pprint(cfg)

    # Initialize Weights & Biases on main process
    if accelerator.is_main_process:
        wandb.init(
            project="MedSliM-pretraining",
            name="test-run-MRNet",
        )

    # Build model
    print("Creating model...")
    model = MoCo(
        embed_dim=cfg["model"]["cobra"]["embed_dim"],
        contrast_dim=cfg["model"]["cobra"]["contrast_dim"],
        input_dims=cfg["model"]["cobra"]["input_dims"],
        num_heads=cfg["model"]["cobra"]["num_heads"],
        num_mamba_layers=cfg["model"]["cobra"]["num_mamba_layers"],
        T=cfg["train"]["temperature"],
        dropout=cfg["model"]["cobra"]["dropout"],
        att_dim=cfg["model"]["cobra"]["attn_dim"],
        d_state=cfg["model"]["cobra"]["mamba_d_state"],
    )

    model_params = sum(p.numel() for p in model.parameters())

    # Learning rate scaling rule 
    base_lr = float(cfg["train"]["learning_rate"])
    global_batch_size = cfg["train"]["batch_size"]
    scaled_lr = base_lr * (global_batch_size / 256.0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=cfg["train"]["weight_decay"])

    max_feature_dim = max(cfg["model"]["cobra"]["input_dims"])
    feat_cfg = cfg["feat_dataset"]
    feat_dirs = feat_cfg["datasets"]
    slice_encoder_models = feat_cfg["model_name"]
    view_planes = args.planes if args.planes else feat_cfg["plane"]
    dataset = PrecomputedFeatPairDataset(
        feat_dirs=feat_dirs,
        slice_encoder_models=slice_encoder_models,
        view_planes=view_planes,
        split="train",
        max_feature_dim=max_feature_dim,
    )

    # Optional: convert to SyncBatchNorm when training across processes for parity with DDP
    if accelerator.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DataLoader
    per_device_batch_size = max(1, int(global_batch_size / max(1, accelerator.num_processes)))
    print(f"batch_size_per_device={per_device_batch_size}")
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        drop_last=True,
        pin_memory=True,
        collate_fn=ssl_collate_fn,
    )

    # Prepare with accelerator
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    
    # Load checkpoint if args.resume is provided
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location="cpu")
        start_epoch = checkpoint.get("epoch", 0)
        accelerator.unwrap_model(model).load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    else:
        if args.resume:
            raise FileNotFoundError(f"No checkpoint found at '{args.resume}'")

    model.train()
    iters_per_epoch = len(loader)
    for e in tqdm(range(start_epoch, cfg["train"]["num_epochs"]), desc="MedSliM Pre-training...", disable=not accelerator.is_main_process):
        total_loss = 0.0

        for i, batch in enumerate(tqdm(loader, leave=False, disable=not accelerator.is_main_process)):
            curr_lr = adjust_learning_rate(optimizer, e + i / iters_per_epoch, scaled_lr, cfg)
            curr_m = adjust_moco_momentum(e + i / iters_per_epoch, cfg)

            x1 = batch["feats1"]
            x2 = batch["feats2"]
            sizes1 = batch["orig_embed_dim1"]
            sizes2 = batch["orig_embed_dim2"]
            seq_lens1 = batch["seq_lens1"]
            seq_lens2 = batch["seq_lens2"]
            x1 = x1.to(dtype=torch.float32)
            x2 = x2.to(dtype=torch.float32)
            sizes1 = sizes1.to(dtype=torch.long)
            sizes2 = sizes2.to(dtype=torch.long)
            seq_lens1 = seq_lens1.to(dtype=torch.long)
            seq_lens2 = seq_lens2.to(dtype=torch.long)
            
            with accelerator.autocast():
                loss = model(x1, x2, input_feature_dims_1=sizes1, input_feature_dims_2=sizes2, 
                           seq_lengths_1=seq_lens1, seq_lengths_2=seq_lens2, m=curr_m)
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            # Gradient clipping to prevent exploding gradients
            accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.detach().item()

        if accelerator.is_main_process:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {e+1}; loss: {avg_loss:.4f}; lr: {curr_lr:.5f}")
            wandb.log({
                "train/epoch": e + 1,
                "train/epoch_loss": avg_loss,
                "train/lr": curr_lr,
            }, step=e + 1)
            if (e + 1) % 50 == 0:
                state = {
                    "epoch": e + 1,
                    "state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    state,
                    os.path.join(
                        cfg["train"]["save_ckpt_path"],
                        f"medslim_test_run_MRNet-epoch{e+1}.pth.tar",
                    ),
                )


def adjust_learning_rate(optimizer, epoch, scaled_base_lr, cfg):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg["train"]["warmup_epochs"]:
        lr = scaled_base_lr * epoch / cfg["train"]["warmup_epochs"]
    else:
        lr = scaled_base_lr * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - cfg["train"]["warmup_epochs"]) / (cfg["train"]["num_epochs"] - cfg["train"]["warmup_epochs"]))
        )
        
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def adjust_moco_momentum(epoch, cfg):
    """Adjust moco momentum based on current epoch"""
    m = 1.0 - 0.5 * (1.0 + math.cos(math.pi * epoch / cfg["train"]["num_epochs"])) * (
        1.0 - cfg["train"]["momentum"]
    )
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSliM-pretraining.")
    parser.add_argument(
        "-c", "--config", type=str, default="../configs/pretrain.yml", help="Path to the config file"
    )
    parser.add_argument(
        "--planes", 
        nargs='*',  
        type=str,   
        help='A list of planes to preprocess.',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="Path to latest checkpoint",
    )
    args = parser.parse_args()
    
    # Get the directory of the current script
    curr_dir = Path(__file__).resolve().parent
    config_path = os.path.join(curr_dir, args.config)

    with open(config_path, "r") as f:
        cfg_data = yaml.safe_load(f)

    template_env = Environment(loader=FileSystemLoader(searchpath="./"))
    template = template_env.from_string(str(cfg_data))
    # Render the template with the values from the config_data
    cfg = yaml.safe_load(template.render(**cfg_data))
    
    # If continual training, use the same directory as the loaded checkpoint; otherwise create a new saved checkpoint folder
    if args.resume:
        save_dir = str(Path(args.resume).parent)
        print(f"Resuming from checkpoint, saving to existing directory: {save_dir}")
    else:
        save_dir = f"{cfg['train']['save_ckpt_path']}/{CURR_TIME}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
    
    # Update config with the resolved save directory
    cfg["train"]["save_ckpt_path"] = save_dir
    
    main(args, cfg)