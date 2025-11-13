#!/usr/bin/env python3
"""
Download a compact subset of CT-RATE from the Hugging Face Hub.

- Pulls volumes only from `dataset/train_fixed/` and `dataset/valid_fixed/`
- Limits to N unique patients per split by parsing folder names like:
  train_53_a_1/, valid_120_b_2/  -> patient_id=53 or 120
- Also downloads label files under `dataset/multi_abnormality_labels/`
- Produces a manifest CSV listing what was downloaded.

Requirements:
  pip install "huggingface_hub[hf_transfer]"

Usage (example):
  python download_CTRate.py \
    --dest /hpcwork/rwth1833/datasets/CT-RATE \
    --train-n 1000 --valid-n 200 \
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import list_repo_tree, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

def parse_args():
    args = argparse.ArgumentParser(description="Download a compact CT-RATE subset.")
    args.add_argument("--dest", default="/hpcwork/rwth1833/datasets/CT-RATE", help="Destination folder")
    args.add_argument("--train-n", type=int, default=1000, help="Max train patients (default: 1000)")
    args.add_argument("--valid-n", type=int, default=200, help="Max valid patients (default: 200)")
    args.add_argument("--ext", default=".nii.gz",
                    help="Volume file extension to download (default: .nii.gz)")
    return args.parse_args()

def _collect_allowlist_and_ids(
    split_name: str,
    max_patients: int,
    ext: str,
    tree,
) -> Tuple[List[str], List[str]]:
    """
    Scan the repo tree and return:
      - allowlist: exact file paths to download for this split
      - patient_ids: sorted unique patient IDs included
    """
    split_prefix = f"dataset/{split_name}/"
    # Match paths like:
    # dataset/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz
    # dataset/valid_fixed/valid_42/valid_42_b/valid_42_b_2.nii.gz
    # Capture groups:
    #  1: 'train' or 'valid'
    #  2: patient id digits
    pat = re.compile(
        rf"{re.escape(split_prefix)}(train|valid)_(\d+)/[^/]+/[^/]+{re.escape(ext)}$"
    )

    selected_patients = set()
    allow = set()
    
    total_nodes = 0
    matching_prefix = 0
    matching_ext = 0

    for node in list(tree):
        total_nodes += 1
        p = getattr(node, "path", None)
        if not p:
            continue
            
        if p.startswith(split_prefix):
            matching_prefix += 1
            if split_name in ["train_fixed", "valid_fixed"] and matching_prefix <= 10:  # Show first 10 matches
                print(f"[DEBUG] Found prefix match: {p}")
        
        if p.endswith(ext):
            matching_ext += 1
            if split_name in ["train_fixed", "valid_fixed"] and matching_ext <= 10:  # Show first 10 matches
                print(f"[DEBUG] Found ext match: {p}")
        
        if not p.startswith(split_prefix) or not p.endswith(ext):
            continue
        
        # Debug: print paths that match prefix and extension
        if split_name in ["train_fixed", "valid_fixed"] and len(allow) < 5:  # Only show first few for debugging
            print(f"[DEBUG] Checking path: {p}")
        
        m = pat.match(p)
        if not m:
            if split_name in ["train_fixed", "valid_fixed"] and len(allow) < 5:
                print(f"[DEBUG] No regex match for: {p}")
            continue
            
        pid = m.group(2)  # digits after train_/valid_
        if split_name in ["train_fixed", "valid_fixed"] and len(allow) < 5:
            print(f"[DEBUG] Matched! Patient ID: {pid}, Path: {p}")
        
        # Admit a new patient if under cap:
        if pid not in selected_patients and len(selected_patients) < max_patients:
            selected_patients.add(pid)
        # Include this file if it belongs to an already selected patient:
        if pid in selected_patients:
            allow.add(p)
    
    print(f"[DEBUG] Total nodes: {total_nodes}, Prefix matches: {matching_prefix}, Ext matches: {matching_ext}")

    return sorted(allow), sorted(selected_patients, key=lambda x: int(x))


def _collect_label_files(tree) -> List[str]:
    """Grab all files under dataset/multi_abnormality_labels/ (CSV/Parquet/etc.)."""
    root = "dataset/multi_abnormality_labels/"
    allow = []
    for node in list(tree):
        p = getattr(node, "path", None)
        if p and p.startswith(root):
            allow.append(p)
    return sorted(allow)


def main():
    args = parse_args()
    os.makedirs(args.dest, exist_ok=True)

    # Speeds up LFS transfers if the optional plugin is installed
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Helper to create a fresh (recursive) generator each time
    def iter_tree():
        return list_repo_tree(
            repo_id="ibrahimhamamci/CT-RATE",
            repo_type="dataset",
            token=HF_TOKEN,
            recursive=True,
        )

    # Build allow-list for each split
    try:
        # allow_train, train_ids = _collect_allowlist_and_ids(
        #     "train_fixed", args.train_n, args.ext, iter_tree()
        # )
        allow_valid, valid_ids = _collect_allowlist_and_ids(
            "valid_fixed", args.valid_n, args.ext, iter_tree()
        )
        allow_labels = _collect_label_files(iter_tree())
    except HfHubHTTPError as e:
        print(f"[ERROR] Failed to list repo tree: {e}", file=sys.stderr)
        sys.exit(1)

    # Combined allowlist (exact paths)
    # allow_patterns = sorted(set(allow_train + allow_valid + allow_labels))
    allow_patterns = sorted(set(allow_valid + allow_labels))
    # print(f"[INFO] Selected patients -> train: {len(train_ids)} | valid: {len(valid_ids)}")
    print(f"[INFO] Selected patients -> valid: {len(valid_ids)}")
    print(f"[INFO] Files to download: {len(allow_patterns)}")

    # Do the filtered snapshot download
    snapshot_path = snapshot_download(
        repo_id="ibrahimhamamci/CT-RATE",
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=args.dest,
        token=HF_TOKEN,
    )
    print(f"[OK] Downloaded subset under: {snapshot_path}")

if __name__ == "__main__":
    main()

