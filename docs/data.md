# Datasets

Knee MRI datasets for self-supervised pretraining and downstream evaluation.

## Directory Structure

After preprocessing, volumes are stored as compressed NIfTI (`.nii.gz`) with shape `(W, H, D)` following the TorchIO convention. All datasets share the same layout:

```
<DatasetName>/
├── <split>/[<sequence>/]<plane>/<uid>.nii.gz
└── <split>.csv
```

The optional `<sequence>/` subfolder is present when a dataset contains multiple MRI sequences (e.g., `pd/`, `t2_fs/` for fastMRI; `DESS_E1/`, `DESS_E2/` for SKM-TEA). Single-sequence datasets (MRNet, kneeMRI, KMAR-50K) omit this level.

## Datasets Overview

| Dataset | Modality | Sequence | Plane | Size | Labels | Download Source |
|---------|----------|----------|-------|------|--------|--------|
| MRNet | Knee MRI | PD / T2 (varies) | Sagittal, Coronal, Axial | 1,130 / 120 (train/test) exams | Abnormal, ACL tear, Meniscal tear | [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mrnet/) |
| fastMRI Knee | Knee MRI | PD, PDFS, T1, T2, T1PRE, T1POST, ... | Sagittal, Coronal, Axial | ~10,000 exams (~164 GB) | None (unlabeled) | [NYU fastMRI](https://fastmri.med.nyu.edu/) |
| KMAR-50K | Knee MRI | Various (clinical routine) | Sagittal, Coronal, Axial | ~50,000 volumes | Available but unused | [Zenodo](https://zenodo.org/records/14993753) |
| SKM-TEA | Knee qDESS MRI | DESS Echo 1 (T1w, TE≈6ms), DESS Echo 2 (T2w, TE≈34ms) | Sagittal | 86 / 43 / 26 (train/val/test) exams | Meniscal tear, Ligament tear, Cartilage lesion, Effusion | [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7) |
| kneeMRI | Knee MRI | Sagittal PD | Sagittal | 817 / 100 (train/test) volumes | ACL tear (binary & multiclass) | [Zenodo](https://doi.org/10.5281/zenodo.3255079) |

MRNet, fastMRI Knee, and KMAR-50K are used for **self-supervised pretraining**. SKM-TEA and kneeMRI are used for **downstream evaluation**.

## Preprocessing

### 1. MRNet.

```bash
python scripts/preprocess_dataset/preprocess_MRNet.py \
    --data-dir /path/to/MRNet/MRNet-v1.0 \
    --save-dir /path/to/datasets/preprocessed/MRNet \
    --workers 8
```

### 2. fastMRI Knee

Preprocessing converts DICOM series to NIfTI, groups by patient/exam/series, and determines the acquisition plane and MRI sequence from DICOM metadata:

```bash
python scripts/preprocess_dataset/preprocess_fastMRI.py \
    --data-dir /path/to/fastMRI/knee \
    --save-dir /path/to/datasets/preprocessed/fastMRI \
    --split train \
    --workers 32
```

### 3. KMAR-50K

```bash
python scripts/preprocess_dataset/preprocess_KMAR-50K.py \
    --data-dir /path/to/KMAR-50K \
    --save-dir /path/to/datasets/preprocessed/KMAR-50K \
    --workers 8
```

### 4. SKM-TEA

```bash
python scripts/preprocess_dataset/preprocess_SKM-TEA.py \
    --data-dir /path/to/SKM-TEA/qdess/v1-release \
    --save-dir /path/to/datasets/preprocessed/SKM-TEA \
    --workers 8
```

### 5. kneeMRI

```bash
python scripts/preprocess_dataset/preprocess_kneeMRI.py \
    --data-dir /path/to/kneeMRI \
    --save-dir /path/to/datasets/preprocessed/kneeMRI \
    --split test \
    --plane sagittal \
    --workers 8
```