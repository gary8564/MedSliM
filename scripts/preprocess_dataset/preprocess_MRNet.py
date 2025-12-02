import logging
import sys
import argparse
import torchio as tio 
import numpy as np 
import pandas as pd
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from pathlib import Path 


logger = logging.getLogger(__name__)

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
    # Read
    data = np.load(path_file)
    
    # In torchio.ScalarImage, the tensor shape should be [C, W, H, D]. But the data shape is [D, H, W].
    # So we need to swap the dimension.
    data = np.swapaxes(data, 0, -1)
    
    # Convert to Nifti 
    img = tio.ScalarImage(tensor=data[None])

    # Write
    file_stem = path_file.stem 
    path_out_dir = save_dir / path_file.parent.relative_to(data_dir)
    path_out_dir.mkdir(parents=True, exist_ok=True)
    img.save(path_out_dir / f'{file_stem}.nii.gz')
    
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
    df_train.to_csv(save_dir/'train.csv', index=False)
    df_val.to_csv(save_dir/'valid.csv', index=False)
    
    logger.info(f"NIfTI files written: {len(list(save_dir.rglob('*.nii.gz')))}")
    logger.info(f"Annotation csv files written: {len(list(save_dir.rglob('*.csv')))}")    
    logger.info(f"Number train.csv: {len(df_train)} of 1130, Number valid.csv: {len(df_val)} of 120")
    for cls in ['abnormal', 'acl', 'meniscus']:
        logger.info(f"{df_train[cls].value_counts(normalize=True)} {cls} labels in train dataset.")
        logger.info(f"{df_val[cls].value_counts(normalize=True)} {cls} labels in valid dataset.")
    logger.info("================================================")
    logger.info("Preprocessing completed.")
    logger.info("================================================")


if __name__ == "__main__":
    main()