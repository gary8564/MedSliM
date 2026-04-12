import argparse
import logging
import re
import sys
import pandas as pd
import pydicom
import torch 
import numpy as np
import pylidc as pl
import torchio as tio 
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path 
from multiprocessing import Pool
from functools import partial

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

def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        # return [maybe_convert(item) for item in x]
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.dataset.Dataset):  
        # return dataset2dict(x)
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x 


def dataset2dict(ds, exclude=['PixelData', '']):
    return {keyword:value for key in ds.keys() 
            if ((keyword := ds[key].keyword) not in exclude)  and ((value := maybe_convert(ds[key].value)) is not None) }


def scan2nifti(scan_id, save_dir):
    # Get DICOMs in correct order (PyLIDC fixes duplicate z and sorts by z coordinate)
    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
    images = scan.load_all_dicom_images()

    # Get path to series
    path_series = Path(scan.get_path_to_dicom_files())
    
    # In-plane pixel spacing (mm) — LIDC uses square pixels
    row_spacing = col_spacing = scan.pixel_spacing
    
    # Through-plane spacing (mm)
    slice_spacing = float(abs(scan.slice_spacing))
    
    # Image position and orientation (row and column direction cosines)
    image_position = np.array(images[0].ImagePositionPatient, float)
    image_orientation = np.array(images[0].ImageOrientationPatient, float)
    row_cosines = image_orientation[:3]   # direction of increasing column index (along a row)
    col_cosines = image_orientation[3:]   # direction of increasing row index (down a column)
    slice_cosines = np.cross(row_cosines, col_cosines)

    # Construct affine in DICOM (LPS) coordinate system.
    # Volume from to_volume() has shape (Rows, Cols, Slices):
    #   axis 0 (row index)    → moves in column direction → col_cosines
    #   axis 1 (column index) → moves in row direction    → row_cosines
    affine = np.eye(4)
    affine[:3, 0] = col_cosines * row_spacing
    affine[:3, 1] = row_cosines * col_spacing
    affine[:3, 2] = slice_cosines * slice_spacing
    affine[:3, 3] = image_position

    # Convert from DICOM LPS to NIfTI RAS: negate x (L→R) and y (P→A)
    affine[0, :] *= -1
    affine[1, :] *= -1

    # Return the scan as a 3D numpy array volume and affine transform to world coordinates
    img_vol = scan.to_volume()
    img_tio = tio.ScalarImage(tensor=img_vol[None], affine=affine)

    # Read Metadata 
    ds = pydicom.dcmread(next(path_series.glob('*.dcm'), None), stop_before_pixels=True)
    metadata = dataset2dict(ds)

    # Write to NIfTI file
    filename = f'{scan_id}.nii.gz'
    logger.info(f"Writing file: {filename}:")
    img_tio.save(save_dir/filename )

    # Add additional information 
    metadata['size'] = list(img_tio.spatial_shape)
    metadata['nifti_path'] = str(save_dir/filename)
    metadata['scan_id'] = scan_id
    return metadata



def main():
    parser = argparse.ArgumentParser(description="Preprocess LIDC-IDRI")
    parser.add_argument("--data-dir", type=str, default="/hpcwork/rwth1833/datasets/LIDC-IDRI/TCIA_LIDC-IDRI_20200921/LIDC-IDRI", help="Root folder containing the LIDC-IDRI dataset")
    parser.add_argument("--save-dir", type=str, default="/hpcwork/rwth1833/datasets/preprocessed/LIDC-IDRI", help="Output directory for NIfTI and metadata")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Logging 
    setup_logging(args.verbose)
    
    # Convert save_dir to Path and create if needed
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get all scans 
    scan_ids = range(1, len(list(pl.query(pl.Scan)))+1)

    # Parallel processing
    metadata_list = []
    scan2nifti_with_dir = partial(scan2nifti, save_dir=save_dir)
    with Pool(processes=max(1, int(args.workers))) as pool:
        for meta in tqdm(pool.imap_unordered(scan2nifti_with_dir, scan_ids), total=len(scan_ids)):
            metadata_list.append(meta)

    # Save metadata 
    df = pd.DataFrame(metadata_list)
    df.to_csv(save_dir / 'metadata.csv', index=False)

    # Check export 
    num_scans = len(list(save_dir.rglob('*.nii.gz')))
    logger.info(f"Finished. NIfTI files written: {num_scans}")

    logger.info("================================================")
    logger.info("Preprocessing completed.")
    logger.info("================================================")

if __name__ == "__main__":
    main()
    
