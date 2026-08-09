# 1. Train-test patient leakage
import pandas as pd
from pathlib import Path

root_path = Path("/hpcwork/rwth1833/datasets/preprocessed")
nod = pd.read_csv(root_path / "LIDC-IDRI-nodules" / "nodule_metadata.csv")
assert len(set(nod[nod.split=="train"].patient_id) & set(nod[nod.split=="test"].patient_id)) == 0

# 2. CV scan leakage (current protocol — EXPECT FAIL)
from sklearn.model_selection import StratifiedKFold
train = nod[nod.split=="train"]
for tr, va in StratifiedKFold(3, shuffle=True, random_state=42).split(train, train.Malignant):
    assert len(set(train.iloc[tr].scan_id) & set(train.iloc[va].scan_id)) == 0  # fails ~190/fold

# 3. MST count reconciliation
ann = pd.read_csv(root_path / "LIDC-IDRI" / "annotation.csv")
key = ["patient_id","study_instance_uid","series_instance_uid","scan_id","nodule_idx"]
mst_n = ann.groupby(key).malignancy.apply(lambda x: int(x.mean().round()))
assert (mst_n != 3).sum() == 1625  
assert len(pd.read_csv(root_path / "nodule_labels.csv")) == 1616  # MedSliM

# 4. Feature-label alignment
assert len(glob(root_path / "feat_caches" / "..." / "train" / "axial" / "*.safetensors")) == 1266