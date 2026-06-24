import logging
import sys
import argparse
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd
import torchio as tio
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

from med_slim.utils.preprocessing.slice_axis_resolver import PLANE_TO_AXIS, build_slice_last_affine

logger = logging.getLogger(__name__)

# Zero-pad IDs to match filenames (e.g. 0 -> 0000)
ID_WIDTH = 4

def _save_slice_last_nifti(
    volume: np.ndarray,
    out_path: Union[str, Path],
    plane: str,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> tio.ScalarImage:
    img = tio.ScalarImage(
        tensor=np.asarray(volume, dtype=np.float32)[None],
        affine=build_slice_last_affine(plane, spacing),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return img


def _extract_plane_from_rel_path(rel_path: Path) -> str:
    """Infer view plane from a path like ``train/sagittal`` or ``valid/coronal``."""
    for part in rel_path.parts:
        if part in PLANE_TO_AXIS:
            return part
    raise ValueError(
        f"Cannot infer plane from {rel_path!r}; "
        f"expected one of {sorted(PLANE_TO_AXIS)} in the path."
    )


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        
def npy2nifti(path_file, save_dir, data_dir):
    data = np.load(path_file)
    
    # In torchio.ScalarImage, the tensor shape should be [C, W, H, D]. But the data shape is [D, H, W].
    # So we need to swap the dimension.
    data = np.swapaxes(data, 0, -1)

    file_stem = path_file.stem
    if file_stem.isdigit():
        file_stem = f"{int(file_stem):0{ID_WIDTH}d}"
    rel_path = path_file.parent.relative_to(data_dir)
    if rel_path.parts[0] == "valid":
        rel_path = Path("test") / Path(*rel_path.parts[1:])
    path_out_dir = save_dir / rel_path
    out_file = path_out_dir / f'{file_stem}.nii.gz'
    plane = _extract_plane_from_rel_path(rel_path)
    _save_slice_last_nifti(data, out_file, plane=plane)
    
def combine_annotation_csv(data_dir, split):
    # Combine different annotation csv files into one
    df = pd.DataFrame()
    for pathology in ['abnormal', 'acl', 'meniscus']:
        df_pathology = pd.read_csv(data_dir/f'{split}-{pathology}.csv', names=['ID', pathology])
        df = pd.merge(df, df_pathology, on='ID') if len(df) > 0 else df_pathology
    return df

def main():
    parser = argparse.ArgumentParser(description="Preprocess MRNet")
    parser.add_argument("--data-dir", type=str, default="/hpcwork/rwth1833/datasets/MRNet/MRNet-v1.0", help="Root folder containing the MRNet dataset")
    parser.add_argument("--save-dir", type=str, default="/hpcwork/rwth1833/datasets/preprocessed/MRNet", help="Output directory for NIfTI and metadata")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    
    logger.info("================================================")
    logger.info("Step 1: Image preprocessing:Convert NPY to NIfTI")
    logger.info("================================================") 
       
    # Convert save_dir to Path and create if needed
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all npy files
    path_files = list(data_dir.rglob('*npy'))

    # Parallel processing
    npy2nifti_with_dir = partial(npy2nifti, save_dir=save_dir, data_dir=data_dir)
    with Pool(processes=max(1, int(args.workers))) as pool:
        for _ in tqdm(pool.imap_unordered(npy2nifti_with_dir, path_files), total=len(path_files)):
            pass
        
    logger.info("================================================")
    logger.info("Step 2: Annotation preprocessing: Combine different annotation csv files into one")
    logger.info("================================================")
    df_train = combine_annotation_csv(data_dir, 'train')
    df_val = combine_annotation_csv(data_dir, 'valid')
    # Zero-pad ID column to match NIfTI filenames (e.g. 0 -> 0000)
    for df in (df_train, df_val):
        df['ID'] = df['ID'].apply(lambda x: f"{int(x):0{ID_WIDTH}d}" if str(x).isdigit() else x)
    df_train.to_csv(save_dir/'train.csv', index=False)
    df_val.to_csv(save_dir/'test.csv', index=False)
    
    logger.info(f"NIfTI files written: {len(list(save_dir.rglob('*.nii.gz')))}")
    logger.info(f"Annotation csv files written: {len(list(save_dir.rglob('*.csv')))}")    
    logger.info(f"Number train.csv: {len(df_train)} of 1130, Number test.csv: {len(df_val)} of 120")
    for cls in ['abnormal', 'acl', 'meniscus']:
        logger.info(f"{df_train[cls].value_counts(normalize=True)} {cls} labels in train dataset.")
        logger.info(f"{df_val[cls].value_counts(normalize=True)} {cls} labels in test dataset.")
    logger.info("================================================")
    logger.info("Preprocessing completed.")
    logger.info("================================================")


if __name__ == "__main__":
    main()