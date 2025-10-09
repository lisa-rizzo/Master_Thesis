import numpy as np
from pathlib import Path
import glare_clean.datasets.VerSe.get_data_VerSe as get_data_VerSe

if __name__ == "__main__":
    dataset_dir = "/DATA/anjany/prepared"
    excel_path = "/home/student/lisa_ma/data_filter_joined.xlsx"

    files, missing = get_data_VerSe.get_filtered_files_across_dsnames(
        root_dir=Path(dataset_dir),
        excel_path=excel_path,
        split="train",
        glob_pattern="*.npz",
        pid_col="pid",
        dsname_col="dsname",
        verts_col="vert_label",
        check_complete=True,
        apply_excel_filter=True,
    )
    print(f"Anzahl Trainingsdateien: {len(files)}")

    # Erster Durchlauf: Mittelwert berechnen
    n_voxels = 0
    sum_voxels = 0.0

    for path in files:
        data = np.load(path)
        img = data['img']
        flat = img.flatten()
        n_voxels += flat.size
        sum_voxels += flat.sum()

    mean = sum_voxels / n_voxels

    # Zweiter Durchlauf: Standardabweichung berechnen
    sum_sq_diff = 0.0
    for path in files:
        data = np.load(path)
        img = data['img']
        flat = img.flatten()
        sum_sq_diff += ((flat - mean) ** 2).sum()

    std = np.sqrt(sum_sq_diff / n_voxels)

    print(f"mean: {mean}")
    print(f"std: {std}")