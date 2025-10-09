from pathlib import Path
import pandas as pd
import ast
from collections import defaultdict

def load_ds_pid_vert_map_from_excel(
    excel_path: str,
    split: str = "train",
    pid_col: str = "pid",
    dsname_col: str = "dsname",
    verts_col: str = "vert_label",
    has_T1_col: str = "has_T1",
    is_consec_col: str = "is_consecutive",
    apply_excel_filter: bool = True, #Applies (has_T1==1) & (is_consecutive==1) & (split==split) if apply_excel_filter is True
) -> dict[str, dict[str, set[int]]]: 
    
    #Returns nested dict: { dsname: { pid: set_of_expected_vertebra_ids } }
   
    df = pd.read_excel(excel_path) 

    if apply_excel_filter:
        df = df.loc[
            (df[has_T1_col] == 1) &
            (df[is_consec_col] == 1) &
            (df["split"] == split)
        ]

    #Initializes nested dictionary: keys: dsnames str
    ds_pid_vert: dict[str, dict[str, set[int]]] = {}
    for _, row in df.iterrows():
        dsname = str(row[dsname_col]).strip()
        pid = str(row[pid_col]).strip()
        verts_val = row[verts_col]

        # Parse vertebra list (string like "[8, 9, 10, ...]" or actual list) Normalizes verts_val into Python list of values:
        if isinstance(verts_val, str):
            verts_list = ast.literal_eval(verts_val)
        elif hasattr(verts_val, "__iter__"):
            verts_list = list(verts_val)
        else:
            verts_list = [verts_val]

        verts_set = set(int(v) for v in verts_list if pd.notna(v))


        if dsname not in ds_pid_vert:
            ds_pid_vert[dsname] = {}
        ds_pid_vert[dsname][pid] = verts_set

    return ds_pid_vert


def extract_pid_and_vert_from_filename(p: Path) -> tuple[str | None, int | None]:
    """
    Filename examples:
      segment0.8mm_vert1_sub-ATL003_dir-sag_split-HWS.npz
      segment0.8mm_vert10_sub-csi003.npz
    Extract:
      pid  = string between '_sub-' and '.npz' (or next separator if present)
             e.g., 'ATL003_dir-sag_split-HWS' or 'csi003'
      vert = integer after 'vert' in the filename, e.g., 1, 10, etc.
    """
    name = p.name
    stem = p.stem  # without .npz

    # Extract vertebra number
    vert = None
    for part in stem.split('_'):
        if part.startswith("vert"): 
            try:
                vert = int(part.replace("vert", ""))
            except ValueError:
                vert = None
            break

    # Extract pid: everything after '_sub-' up to end (or up to next known marker if present)
    pid = None
    if "_sub-" in stem:
        # Get everything after '_sub-'
        after_sub = stem.split("_sub-")[1]
        # In ATL files, this continues to include dir/split info; we should keep it as-is,
        # because Excel pid has the same full token (e.g., 'ATL003_dir-sag_split-HWS').
        pid = after_sub  # keep full remainder (without .npz)
    # If no '_sub-' present, try an alternate pattern (not expected for your files)
    return pid, vert



def get_filtered_files_across_dsnames(
    root_dir: str | Path,
    excel_path: str,
    split: str = "train",
    glob_pattern: str = "*.npz",
    pid_col: str = "pid",
    dsname_col: str = "dsname",
    verts_col: str = "vert_label",
    check_complete: bool = True,
    apply_excel_filter: bool = True,
):
    """
    Scans all dsname subdirectories under root_dir, enters <dsname>/<split>/,
    collects .npz files whose (pid, vert) match Excel constraints.
    Returns:
      files: list[Path]
      missing: dict[dsname][pid] -> sorted list of missing vertebra ids
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    ds_pid_vert_map = load_ds_pid_vert_map_from_excel(
        excel_path=excel_path,
        split=split,
        pid_col=pid_col,
        dsname_col=dsname_col,
        verts_col=verts_col,
        apply_excel_filter=apply_excel_filter,
    )

    collected: list[Path] = []
    missing_summary: dict[str, dict[str, list[int]]] = {}

    # Iterate dsname directories known from Excel map (so we only search needed ones)
    for dsname, pid_vert_map in ds_pid_vert_map.items():
        ds_split_dir = root / dsname / split
        if not ds_split_dir.exists():
            # Skip silently if directory is not present on disk
            continue

        # Collect matching files
        found_verts_per_pid: defaultdict[str, set[int]] = defaultdict(set)
        for p in ds_split_dir.glob(glob_pattern):
            if p.suffix.lower() != ".npz":
                continue
            pid, vert = extract_pid_and_vert_from_filename(p)
            if pid is None or vert is None:
                continue
            # EXACT match with Excel pid (full token after sub-)
            if pid in pid_vert_map and vert in pid_vert_map[pid]:
                collected.append(p)
                found_verts_per_pid[pid].add(vert)

        # Completeness check per pid for this dsname
        if check_complete:
            for pid, expected_verts in pid_vert_map.items():
                missing = expected_verts - found_verts_per_pid.get(pid, set())
                if missing:
                    missing_summary.setdefault(dsname, {})[pid] = sorted(missing)

    return collected, missing_summary
    


if __name__ == "__main__":
    
    root_dir = "/DATA/anjany/prepared"
    excel_file = "/home/student/lisa_ma/data_filter_joined.xlsx"
    split = "train"

    files, missing = get_filtered_files_across_dsnames(
        root_dir=root_dir,
        excel_path=excel_file,
        split=split,
        glob_pattern="*.npz",
        pid_col="pid",
        dsname_col="dsname",
        verts_col="vert_label",
        check_complete=True,
        apply_excel_filter=True,
    )

    print(f"Collected {len(files)} matching NPZ files.")
    if missing:
        print("Missing vertebrae per PID per dataset:")
        for ds, mp in missing.items():
            print(f"Dataset: {ds}")
            for pid, verts in mp.items():
                print(f"  PID {pid}: missing verts {verts}")
    else:
        print("All expected vertebra files present.")


    print("vert + pid ", extract_pid_and_vert_from_filename(Path("segment0.8mm_vert15_sub-ATL089_dir-sag_split-BWSLWS.npz")))
   


    
