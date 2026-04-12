import argparse
import logging
import re
import sys
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool


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
        
def load_filepath_mapping_dataframe(mapping_path: Path, data_dir: Path) -> pd.DataFrame:
    expected_cols = ["original_path_and_filename", "descriptive_path", "classic_path"]
    if mapping_path.suffix.lower() in [".xlsx", ".xls"]:
        df_map_raw = pd.read_excel(mapping_path)
        df_map_filtered = df_map_raw[expected_cols]
        logger.info("Normalizing descriptive paths and saving corrected CSV ...")
        df_map = correct_mapping_dataframe(df_map_filtered, data_dir / "corrected-Breast-Cancer-MRI-filepath_filename-mapping.csv")
    else:
        df_map = pd.read_csv(mapping_path)
        df_map = df_map[expected_cols]
    # Ensure required columns exist
    missing = [c for c in expected_cols if c not in df_map.columns]
    if missing: 
        raise ValueError(f"Mapping file missing required columns: {missing}. Available: {list(df_map.columns)}")
    return df_map

def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        return None
    elif isinstance(x, pydicom.dataset.Dataset):
        return None
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x

def dataset2dict(ds, exclude=['PixelData', '']):
    result = {}
    for key in ds.keys():
        kw = ds[key].keyword
        if kw in exclude:
            continue
        val = maybe_convert(ds[key].value)
        if val is not None:
            result[kw] = val
    return result

def get_patient_id_from_path(path_str: str) -> str:
    m = re.search(r"Breast_MRI_(\d+)", path_str)
    if m:
        return m.group(1)
    return "unknown"

def get_dicom_dir(dataset_root: Path, dicom_filepath: str) -> Path:
    # Get the parent directory of dicom files for ImageSeriesReader
    p = Path(dicom_filepath)
    if not p.is_absolute():
        p = (dataset_root / p).resolve()
    return p.parent

def normalize_descriptive_path(path_str: str) -> str:
    # Normalize known discrepancies between path in mapping table and actual filesystem
    s = str(path_str)
    # BreastMRI### → Breast_MRI_### when used as a segment
    s = re.sub(r'(^|/)BreastMRI(\d+)(?=/)', r'\1Breast_MRI_\2', s)
    # Remove carets inside folder names like BREAST^ROUTINE → BREASTROUTINE
    s = s.replace('^', '')
    s = s.replace('+', '')
    # Fix phase token splitting: "-Ph1/ax" → "-Ph1ax" (also Ph2/Ph3)
    s = re.sub(r'-Ph(\d)\s*/\s*ax', r'-Ph\1ax', s)
    s = re.sub(r'-Ph(\d)\s*/\s*Ax', r'-Ph\1Ax', s)
    # Contrast wording variants
    s = re.sub(r'W\s*/\s*WO', 'WWO', s)
    s = re.sub(r'W\s*\+\s*W\s*/\s*O', 'W  WO', s)
    s = s.replace('&', '')
    s = re.sub(r'W\s*&\s*WO', 'W  WO', s)
    s = re.sub(r'W\s*/\s*&\s*WO', 'W  WO', s)
    s = re.sub(r'W/\s*WO', 'W  WO', s)
    s = re.sub(r'\bW\s*/\s*O\b', 'WO', s)
    return s

def correct_mapping_dataframe(df_map: pd.DataFrame, save_csv: Path | None = None) -> pd.DataFrame:
    df = df_map.copy()
    if 'descriptive_path' in df.columns:
        df['descriptive_path'] = df['descriptive_path'].astype(str).apply(normalize_descriptive_path)
    if save_csv is not None:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
    return df

def read_dicom_file(dicom_dir: Path):
    dicom_file = next(dicom_dir.glob("*.dcm"), None)
    if dicom_file is None:
        # Fallback: TCIA sometimes ships files without .dcm extension
        dicom_file = next((p for p in dicom_dir.iterdir() if p.is_file()), None)
    if dicom_file is None:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    return pydicom.dcmread(str(dicom_file), stop_before_pixels=True)

def extract_label_from_original_path(original_path: str) -> str:
    # Example: DICOM_Images/Breast_MRI_211/post_1/Breast_MRI_211_post_1_005.dcm -> 'post_1'
    label = Path(original_path).parent.name.lower()
    return label if label in {"pre", "post_1"} else ""

def extract_study_uid_from_classic_path(classic_path: str) -> str:
    # Example classic path: Duke-Breast-Cancer-MRI/Breast_MRI_001/<StudyUID>/<SeriesUID>/1-004.dcm
    return classic_path.split('/')[2]

def build_series_index(dataset_root: Path, df_map: pd.DataFrame) -> pd.DataFrame:
    # Add patient and modality from original path and reduce to one row per series directory
    df = df_map.copy()
    df["patient_id"] = df["original_path_and_filename"].apply(get_patient_id_from_path)
    df["modality"] = df["original_path_and_filename"].apply(extract_label_from_original_path)
    df = df[df["modality"].isin(["pre", "post_1"])].copy()
    df["original_dir"] = df["original_path_and_filename"].apply(lambda p: str(Path(p).parent))
    df["classic_dir"] = df["classic_path"].apply(lambda p: str(Path(p).parent) if isinstance(p, str) else "")
    df["study_uid"] = df["classic_path"].apply(lambda p: extract_study_uid_from_classic_path(p) if isinstance(p, str) else "")

    # Resolve DICOM series dirs
    df["dicom_dir"] = df["descriptive_path"].apply(lambda p: str(get_dicom_dir(dataset_root, p)))
    
    # Deduplicate by patient, modality, original_dir (one series per directory)
    df_series = df.sort_values(["patient_id", "modality", "dicom_dir"]).drop_duplicates(
        subset=["patient_id", "modality", "dicom_dir"], keep="first"
    )
    return df_series

def convert_series_to_nifti(task):
    reader = sitk.ImageSeriesReader()
    reader.SetSpacingWarningRelThreshold(0.01)
    dicom_dir = Path(task["dicom_dir"])
    save_dir = Path(task["save_dir"])

    try:
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        if not dicom_names:
            raise RuntimeError("No series files discovered via GDCM in directory")
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
        # Build output path: /save_dir/Breast_MRI_<id>/<modality>/<modality>.nii.gz
        out_dir = save_dir / f"Breast_MRI_{task['patient_id']}" / task["modality"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{task['modality']}.nii.gz"
        sitk.WriteImage(img, str(out_file))

        # metadata
        ds = read_dicom_file(dicom_dir)
        md = dataset2dict(ds)
        md.update({
            "nifti_path": str(out_file),
            "dicom_dir": str(dicom_dir),
            "modality": task["modality"],
            "patient_id": task["patient_id"],
            "study_uid": task.get("study_uid", ""),
        })
        return md
    except Exception as e:
        logger.warning(f"Failed conversion for {dicom_dir}: {e}")
        return None

def substract_post_contrast_from_pre_contrast(path_patient):
    logger.debug(f"Subtracting post-contrast from pre-contrast for patient {path_patient.name}")
    # Subtracting the first post-contrast images from the pre-contrast images to enhance contrast differences indicative of malignancy versus benign tissue changes in breast cancer.
    pre_contrast_nii = sitk.ReadImage(str(path_patient/'t1_pre'/'t1_pre.nii.gz'), sitk.sitkInt16)
    post_contrast_nii = sitk.ReadImage(str(path_patient/'t1_post_1'/'t1_post_1.nii.gz'), sitk.sitkInt16)
    pre_contrast = sitk.GetArrayFromImage(pre_contrast_nii)
    post_contrast = sitk.GetArrayFromImage(post_contrast_nii)
    sub = post_contrast - pre_contrast
    sub = sub - sub.min() # Note: negative values causes overflow when using uint 
    sub = sub.astype(np.uint16)
    sub_nii = sitk.GetImageFromArray(sub)
    sub_nii.CopyInformation(pre_contrast_nii)
    sitk.WriteImage(sub_nii, str(path_patient/'t1_subtracted.nii.gz'))

def main():
    parser = argparse.ArgumentParser(description="Preprocess Duke Breast MRI: T1 pre and first post to NIfTI")
    parser.add_argument("--data-dir", type=str, default="/hpcwork/rwth1833/datasets/DukeBreastMRI/Duke-Breast-Cancer-MRI_v2_20220609", help="Root folder containing the extracted NBIA files")
    parser.add_argument("--mapping-file", type=str, default="/hpcwork/rwth1833/datasets/DukeBreastMRI/Duke-Breast-Cancer-MRI_v2_20220609/Breast-Cancer-MRI-filepath_filename-mapping.xlsx", help="File path mapping tables	")
    parser.add_argument("--save-dir", type=str, default="/hpcwork/rwth1833/datasets/preprocessed/DukeBreastMRI", help="Output directory for NIfTI and metadata")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    # Avoid oversubscription: let Pool handle parallelism; make ITK single-threaded per worker
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    
    out_dir = Path(args.save_dir)
    if not out_dir.is_absolute():
        out_dir = (Path(args.data_dir) / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("================================================")
    logger.info("Step 1: Convert DICOM to NIfTI")
    logger.info("================================================")
    logger.info("Loading filepath mapping table...")
    df_map = load_filepath_mapping_dataframe(Path(args.mapping_file), Path(args.data_dir))
    logger.info("Building index from filepath mapping table...")
    df_series = build_series_index(Path(args.data_dir), df_map)
    logger.info(f"Indexed unique series: {len(df_series)}")

    modality_to_target = {"pre": "t1_pre", "post_1": "t1_post_1"}
    tasks = []
    for _, row in df_series.iterrows():
        tasks.append({
            "dicom_dir": row["dicom_dir"],
            "save_dir": str(out_dir),
            "modality": modality_to_target.get(row["modality"], row["modality"]),
            "patient_id": row["patient_id"],
            "study_uid": row.get("study_uid", ""),
        })

    logger.info("Converting DICOM to NIfTI in parallel ...")
    metadata_rows = []
    with Pool(processes=max(1, int(args.workers))) as pool:
        # Use a chunk size to reduce scheduling overhead
        chunksize = max(1, len(tasks) // (max(1, int(args.workers)) * 4))
        for md in tqdm(pool.imap_unordered(convert_series_to_nifti, tasks, chunksize=chunksize), total=len(tasks)):
            if md is not None:
                metadata_rows.append(md)

    df_meta = pd.DataFrame(metadata_rows)
    df_meta.to_csv(out_dir / "metadata.csv", index=False)

    # Check nifti files export
    num_series = len(list(out_dir.rglob("*.nii.gz")))
    logger.info(f"Finished. NIfTI files written: {num_series}")
    
    logger.info("================================================")
    logger.info("Step 2: Subtract post-contrast from pre-contrast")
    logger.info("================================================")
    logger.info("Subtracting post-contrast from pre-contrast in parallel ...")
    patient_dirs = [p for p in out_dir.iterdir() if p.is_dir()] # list of patient directories
    with Pool(processes=max(1, int(args.workers))) as pool:
        # Use a chunk size to reduce scheduling overhead
        chunksize = max(1, len(patient_dirs) // (max(1, int(args.workers)) * 4))
        for _ in tqdm(pool.imap_unordered(substract_post_contrast_from_pre_contrast, patient_dirs, chunksize=chunksize), total=len(patient_dirs)):
            pass


if __name__ == "__main__":
    main()



