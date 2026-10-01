import time
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import numpy as np
import data.get_data_VerSe as get_data_VerSe 
from pathlib import Path
import re
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
from collections import defaultdict
from lightning_fabric.utilities.seed import pl_worker_init_function
from monai.data.utils import set_rnd
from helper import paths



RNG_SEEDING = "worker_init+set_rnd v1"


def seed_worker(worker_id: int) -> None:
    """
    worker_init_fn for the training DataLoader.

    Why: MONAI Rand*-transforms draw from their own np.random.RandomState (self.R),
    seeded from OS entropy at Compose construction time — seed_everything does not reach it.
    Without this, two runs with the same seed see different augmentations, all workers draw
    the same stream (forked copies), and every epoch repeats it.

    1. pl_worker_init_function: Lightning only attaches its own worker-seeding function when
       worker_init_fn is None — with a custom one it must run explicitly here.
    2. set_rnd: seeds all Randomizables of the dataset (VerSeDataset.transform -> ImageTransform
       -> Compose) with worker_info.seed = base_seed + worker_id. base_seed is drawn by torch
       per iterator from the main process global RNG: reproducible, distinct per worker and epoch.
       The main process shuffle stream is not touched.

    Not monai.utils.set_determinism: that also sets cudnn.deterministic=True.
    Module-level so the function stays picklable with start_method=spawn.
    """
    pl_worker_init_function(worker_id)
    info = torch.utils.data.get_worker_info()
    set_rnd(info.dataset, seed=info.seed)


# Label spaces. "region" is the 4-class scheme; "vertebra" predicts the vertebra number
# directly (label = number - 1, i.e. 0..N-1).
#
# Why the second mode exists: a 4-class region model cannot in principle detect a counting
# error that does not cross a region boundary. In the first full run, those were exactly the
# ~75 missed cases (mostly +/-1 within a region), while all 69 cross-boundary cases were found.
#
# "region5" adds a dedicated T13 class between T and L (C / T / T13 / L / S).
# In the 4-class space a T13 maps to "thoracic" (or, after Rule A, to L1 in "lumbar")
# and scores like a clean sample in GLARE; in the 25-class space it is not representable.
LABEL_MODES = ("region", "vertebra", "region5")

# Highest vertebra number in the corrected training set (measured: 1..25).
MAX_VERT = 25

# Class names per label space (index = label). vertebra is not named.
REGION_NAMES = ("cervical", "thoracic", "lumbar", "sacral")
REGION5_NAMES = ("cervical", "thoracic", "T13", "lumbar", "sacral")

# Label -1 in an override file means: remove the sample from the dataset.
REMOVE_LABEL = -1


def load_label_overrides(path) -> dict:
    """
    Read an override file (CSV with columns id, label) -> {sample_id: label}.

    Relative paths are resolved against the repo root so that a spec.json containing
    "configs/overrides/...csv" works on any machine.
    """
    import pandas as pd
    path = Path(path)
    if not path.is_absolute():
        path = paths.REPO_ROOT / path
    df = pd.read_csv(path)
    if not {"id", "label"} <= set(df.columns):
        raise ValueError(f"{path}: override file must have columns id,label (got {list(df.columns)})")
    if df["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate ids {df.loc[df['id'].duplicated(), 'id'].tolist()[:5]}")
    return dict(zip(df["id"].astype(str), df["label"].astype(int)))


class VerSeDataset(Dataset):
    def __init__(self, file_paths, transform=None, label_mode: str = "region"):
        """
        file_paths: list of Path objects to .npz files containing 3D grayscale images
        transform: optional transform to apply on the 3D image numpy array (e.g. normalization, augmentation)
        label_mode: "region" (4 classes, default), "vertebra" (vertebra number - 1), or
                    "region5" (C/T/T13/L/S)
        """
        if label_mode not in LABEL_MODES:
            raise ValueError(f"label_mode must be one of {LABEL_MODES}, not {label_mode!r}")
        self.file_paths = file_paths
        self.transform = transform
        self.label_mode = label_mode
        self.pid_corrections = {}
        # {sample_id: label} — overrides the label derived from the filename.
        self.label_overrides = {}

    @staticmethod
    def get_vertebra_label_from_vert(vert: int) -> int:
        """Vertebra number -> 0-based label. -1 for anything outside 1..MAX_VERT."""
        if 1 <= vert <= MAX_VERT:
            return vert - 1
        return -1

    def label_for_vert(self, vert: int) -> int:
        if self.label_mode == "vertebra":
            return self.get_vertebra_label_from_vert(vert)
        if self.label_mode == "region5":
            return self.get_region5_label_from_vert(vert)
        return self.get_region_label_from_vert(vert)

    @staticmethod
    def get_region5_label_from_vert(vert: int) -> int:
        """
        0 = cervical, 1 = thoracic (T1-T12), 2 = T13 (VerSe code 28), 3 = lumbar, 4 = sacral.
        The number used here is the *anatomical* one (see base_label_for), not the corrected one.
        """
        if vert == 28:
            return 2
        r = VerSeDataset.get_region_label_from_vert(vert)
        return r + 1 if r >= 2 else r

    def base_label_for(self, path) -> int:
        """
        Label from the filename, before any override.

        region5: Rule A (28 and 20 present) renumbers T13 to 20 for counting, shifting L1-L5
        to 21-25 — so T13 becomes L1 and L5 becomes S1. For region assignment with a dedicated
        T13 class, the *native* number is correct (28 = T13, 20-24 = L1-L5). The sample_id stays
        the corrected one so all downstream joins (GT, GLARE-CSV) work unchanged.
        """
        if self.label_mode == "region5":
            orig = self.get_vert_number(path)
            mapping = self.pid_corrections.get(self.get_pid_from_path(path), {})
            vert = orig if mapping.get(28) == 20 else self.corrected_vert_for(path)
            return self.get_region5_label_from_vert(vert)
        return self.label_for_vert(self.corrected_vert_for(path))

    def label_for(self, path) -> int:
        """The label that __getitem__ returns (including any override), without loading the file."""
        return self.label_overrides.get(self.sample_id_for(path), self.base_label_for(path))

    @staticmethod
    def num_classes_for(label_mode: str) -> int:
        if label_mode == "vertebra":
            return MAX_VERT
        if label_mode == "region5":
            return len(REGION5_NAMES)
        return len(REGION_NAMES)

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def load_img(path, retries: int = 4, backoff: float = 0.5):
        """
        Load the 'img' array from an npz, with retries.

        Data lives on an NFS mount where occasional incomplete reads occur, reported by numpy
        as `zipfile.BadZipFile: Bad CRC-32 for file 'img.npy'`. This is rare (~1 in 115,000
        reads, observed after 6 clean epochs over the same file set), but without retries a
        single such error aborts the entire run — that is exactly how the first 20-epoch run
        died in epoch 6.

        For GLARE this is even more critical: a full run reads 19,266 samples x 20
        checkpoints = 385,320 times, making such an error practically certain.

        The npz is opened as a context manager; previously the file handle stayed open
        until garbage collection.
        """
        last_err = None
        for attempt in range(retries):
            try:
                with np.load(path) as data:
                    return data['img']
            except Exception as e:   # BadZipFile, OSError, ...
                last_err = e
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
        raise RuntimeError(
            f"Could not read {path} after {retries} attempts (last error: {last_err!r}). "
            f"If this is reproducible, the file is genuinely corrupt rather than a "
            f"transient NFS error."
        )

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image_3d = self.load_img(path)  # shape (D, H, W), grayscale
        image_tensor = torch.tensor(image_3d, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        sample_id = self.sample_id_for(path)
        label = self.label_overrides.get(sample_id, self.base_label_for(path))
        return image_tensor, label, sample_id

    def corrected_vert_for(self, path) -> int:
        """Vertebra number after applying the patient-specific correction."""
        orig_vert = self.get_vert_number(path)
        pid = self.get_pid_from_path(path)
        if pid in self.pid_corrections:
            return self.pid_corrections[pid].get(orig_vert, orig_vert)
        return orig_vert

    def sample_id_for(self, path) -> str:
        """The join key for this sample, without loading the file.

        Identical to what __getitem__ returns as its third element — both go through
        corrected_vert_for() so they can never diverge. Used to filter a subset by id
        without reading 19,266 volumes from disk.
        """
        return f"{self.get_pid_from_path(path)}_vert{self.corrected_vert_for(path)}"
    
    @staticmethod
    def get_vert_number(path):
        path_str = str(path)
        match = re.search(r'_vert(\d+)', path_str)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                    return -1
        else:
            # if no label is found
            return -1
        
    @staticmethod
    def get_region_label_from_vert (vert: int) -> int:
        """
        Assigns region label based on vertebra number:
        0 = cervical (C1-C7)
        1 = thoracic (T1-T12)
        2 = lumbar (L1-L5)
        3 = sacral (S1-S5)
        -1 = unknown/other
        """
        if 1 <= vert <= 7:
            return 0  # cervical
        elif 8 <= vert <= 19:
            return 1  # thoracic
        elif 20 <= vert <= 24:
            return 2  # lumbar
        elif 25 <= vert <= 29:
            return 3  # sacral
        else:
            return -1  # unknown/other
        
    @staticmethod
    def get_pid_from_path(path):
        path_str = str(path)
        # Extract everything after '_sub-' and before '.npz'
        match = re.search(r'_sub-([^.]+)', path_str)
        if match:
            return match.group(1)
        else:
            return "unknown"

def pad_to_max_size(tensors):
    # get max dimension (D, H, W) (channel 1)
    max_depth = max(t.shape[1] for t in tensors)
    max_height = max(t.shape[2] for t in tensors)
    max_width = max(t.shape[3] for t in tensors)

    padded_tensors = []
    for t in tensors:
        d, h, w = t.shape[1], t.shape[2], t.shape[3]
        pad_d = max_depth - d
        pad_h = max_height - h
        pad_w = max_width - w
        # Padding in order (W_right, W_left, H_bottom, H_top, D_back, D_front)
        # We pad the back, bottom, and right sides
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        padded = F.pad(t, padding, mode='constant', value=0)
        padded_tensors.append(padded)
    return padded_tensors


def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    pids = [item[2] for item in batch]

    images = pad_to_max_size(images)
    images = torch.stack(images)  # now all shapes are the same, can stack

    labels = default_collate(labels)  # batch tensor

    return images, labels, pids

def summarize_pid_corrections(pid_corrections: dict) -> dict:
    """
    Count the *actual* renumberings.

    Note: when no pid_fix_* lists are in the spec, the target filter in _compute_pid_corrections
    is off and every PID gets an identity mapping. `len(pid_corrections)` is then the total
    number of patients (1405), not the number of corrections (17). This function only counts
    entries where something actually changes.
    """
    changed, n_vert, n_label = {}, 0, 0
    region = VerSeDataset.get_region_label_from_vert
    for pid, mapping in pid_corrections.items():
        flips = {v: c for v, c in mapping.items() if c != v}
        if not flips:
            continue
        label_flips = {v: (region(v), region(c)) for v, c in flips.items() if region(c) != region(v)}
        changed[pid] = {"verts": flips, "label_flips": label_flips}
        n_vert += len(flips)
        n_label += len(label_flips)
    return {
        "pids": changed,
        "n_pids": len(changed),
        "n_entries_total": len(pid_corrections),
        "n_vert_renumbered": n_vert,
        "n_label_flips": n_label,
    }


def print_pid_correction_summary(pid_corrections: dict, max_pids: int = 20) -> dict:
    """Print the actual corrections (not the identity entries)."""
    s = summarize_pid_corrections(pid_corrections)
    print(
        f"PID corrections: {s['n_pids']} patients affected, "
        f"{s['n_vert_renumbered']} vertebrae renumbered, "
        f"{s['n_label_flips']} region labels changed "
        f"({s['n_entries_total']} entries total, remainder are identity mappings)."
    )
    for i, (pid, info) in enumerate(sorted(s["pids"].items())):
        if i >= max_pids:
            print(f"  ... and {s['n_pids'] - max_pids} more")
            break
        flips = ", ".join(f"{v}->{c}" for v, c in sorted(info["verts"].items()))
        print(f"  {pid}: {flips}")
        for v, (a, b) in sorted(info["label_flips"].items()):
            print(f"      vert {v}: region label {a} -> {b}")
    return s


class VerSeDataLoader(pl.LightningDataModule):
    def __init__(self, spec: dict, train_transforms=None, val_transforms=None, num_workers=8):
        super().__init__()
        self.spec = spec
        self.label_mode = spec.get("label_mode", "region")
        self.num_classes = VerSeDataset.num_classes_for(self.label_mode)
        self.root_dir = Path(spec.get("dataset_dir", paths.DATASET_DIR))
        self.excel_file = spec.get("excel_path", paths.EXCEL_FILTER)
        # LabelOverride column from the GT Excel (A_CT_ANOMALY_LABELv9.xlsx): the radiologist-corrected
        # label sequence for a full scan. Opt-in so that old spec.json files run unchanged.
        self.gt_label_override = bool(spec.get("gt_label_override", False))
        anomaly = Path(spec.get("anomaly_excel_path") or paths.GT_EXCEL)
        self.anomaly_excel_path = anomaly if anomaly.is_absolute() else paths.REPO_ROOT / anomaly
        #self.split = spec.get("split", "train")
        self.batch_size = spec.get("batch_size", 1)
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.train_dataset = None
        self.val_dataset = None


    def _compose_override_with_rules(self, lo_step1_map: dict[int, int]) -> dict[int, int]:
        """
        Step 2: given disk->override mapping from LabelOverride, inspect the resulting
        override vert set for Rule A (vert28+vert20 present) or Rule B (vert20 present,
        vert19 absent, vert18 present) and compose into a single disk->final mapping.
        """
        override_verts = set(lo_step1_map.values())

        rule_map: dict[int, int] = {}
        if 28 in override_verts and 20 in override_verts:          # Rule A: T13 present
            for v in override_verts:
                if v == 28:
                    rule_map[v] = 20
                elif v >= 20:
                    rule_map[v] = v + 1
                else:
                    rule_map[v] = v
        elif (20 in override_verts and
              19 not in override_verts and
              18 in override_verts):                                 # Rule B: missing T12
            for v in override_verts:
                rule_map[v] = v - 1 if v >= 20 else v
        else:
            for v in override_verts:
                rule_map[v] = v

        return {disk_v: rule_map.get(ov, ov) for disk_v, ov in lo_step1_map.items()}

    def _apply_gt_label_override(self, pid_corrections_all: dict) -> None:
        """
        Apply LabelOverride (Step 1 positional + Step 2 Rule A/B) into pid_corrections_all.

        For affected PIDs the LO mapping REPLACES the structural correction (dict.update,
        as in the original). Consequence: sample_ids of these scans change (pid_vert<corrected>),
        and the known cross-region counting errors are corrected before training.

        Note for region5: its Rule-A detection (mapping.get(28) == 20) does not see a T13
        introduced via LO — there the disk number is not 28. Irrelevant for "region".
        """
        lo_step1 = get_data_VerSe.load_label_override_step1(
            filter_excel_path=self.excel_file,
            anomaly_excel_path=self.anomaly_excel_path,
        )
        region = VerSeDataset.get_region_label_from_vert
        changed = {}
        for pid, step1_map in lo_step1.items():
            composed = self._compose_override_with_rules(step1_map)
            before = pid_corrections_all.get(pid, {})
            diff = {d: (before.get(d, d), c) for d, c in composed.items() if before.get(d, d) != c}
            if diff:
                changed[pid] = {
                    "verts": diff,
                    "region_flips": {d: (region(a), region(b)) for d, (a, b) in diff.items()
                                     if region(a) != region(b)},
                }
            pid_corrections_all[pid] = composed
        self.label_override_gt_summary = {
            "anomaly_excel": str(self.anomaly_excel_path),
            "n_pids_with_override": len(lo_step1),
            "n_pids_changed": len(changed),
            "n_vert_renumbered": sum(len(v["verts"]) for v in changed.values()),
            "n_region_flips": sum(len(v["region_flips"]) for v in changed.values()),
            "pids": changed,
        }
        s = self.label_override_gt_summary
        print(f"GT LabelOverride: {s['n_pids_with_override']} PIDs with override, of which "
              f"{s['n_pids_changed']} changed vs. structural correction "
              f"({s['n_vert_renumbered']} vertebrae renumbered, {s['n_region_flips']} region flips; "
              f"incl. PIDs outside train/val)")
        for pid, info in sorted(changed.items()):
            flips = ", ".join(f"{d}: {a}->{b}" for d, (a, b) in sorted(info["region_flips"].items()))
            print(f"  {pid}: region flips {{{flips}}}")

    def _compute_pid_corrections(self, file_paths: list[Path],
                                 target_pids_1820: list | None = None,
                                 target_pids_28: list | None = None) -> dict:
        """
        Only compute corrections for the PIDs listed in target_pids_* if these are provided.
        Rules:
          - Rule A (28 present): if 28 in verts AND 20 in verts:
                map 28 -> 20 and map existing verts >=20 (except 28) -> v+1
          - Rule B (missing 19): if 20 in verts and 19 not in verts and 18 in verts:
                map verts >=20 -> v-1
          - Otherwise identity mapping.
        Returns dict: pid -> { orig_vert: corrected_vert, ... }
        """
        use_target_filter = (target_pids_1820 is not None) or (target_pids_28 is not None)
        target_set = set()
        if target_pids_1820:
            target_set.update(target_pids_1820)
        if target_pids_28:
            target_set.update(target_pids_28)

        pid_to_verts = defaultdict(set)
        # If filtering by target PIDs, only gather entries for them
        for p in file_paths:
            pid = VerSeDataset.get_pid_from_path(p)
            if use_target_filter and pid not in target_set:
                continue
            vert = VerSeDataset.get_vert_number(p)
            if vert >= 0:
                pid_to_verts[pid].add(vert)

        # If no filter specified, ensure we still build mapping for all pids
        if not use_target_filter:
            for p in file_paths:
                pid = VerSeDataset.get_pid_from_path(p)
                vert = VerSeDataset.get_vert_number(p)
                if vert >= 0:
                    pid_to_verts[pid].add(vert)

        pid_corrections = {}
        for pid, verts in pid_to_verts.items():
            verts_set = set(verts)
            mapping = {}
            # Rule A: 28 present and 20 present
            if 28 in verts_set and 20 in verts_set:
                for v in verts_set:
                    if v == 28:
                        mapping[v] = 20
                    elif v >= 20 and v != 28:
                        mapping[v] = v + 1
                    else:
                        mapping[v] = v
            # Rule B: 20 present, 19 missing, 18 present -> shift >=20 down by 1
            elif 20 in verts_set and 19 not in verts_set and 18 in verts_set:
                for v in verts_set:
                    if v >= 20:
                        mapping[v] = v - 1
                    else:
                        mapping[v] = v
            else:
                for v in verts_set:
                    mapping[v] = v
            pid_corrections[pid] = mapping

        return pid_corrections

    def setup(self, stage=None):
        # Train files
        files, missing = get_data_VerSe.get_filtered_files_across_dsnames(
            root_dir=self.root_dir,
            excel_path=self.excel_file,
            split="train",
            glob_pattern="*.npz",
            pid_col="pid",
            dsname_col="dsname",
            verts_col="vert_label",
            check_complete=True,
            apply_excel_filter=True,
        )
        print(f"Collected {len(files)} training files.")
        if missing:
            print("Missing vertebrae per PID per dataset:")
            for ds, mp in missing.items():
                print(f"Dataset: {ds}")
                for pid, verts in mp.items():
                    print(f"  PID {pid}: missing verts {verts}")
        else:
            print("All expected vertebra files present.")

        self.files = files  

        # Val files (collect early so we can compute corrections across both splits)
        val_files, _ = get_data_VerSe.get_filtered_files_across_dsnames(
            root_dir=self.root_dir,
            excel_path=self.excel_file,
            split="val",
            glob_pattern="*.npz",
            pid_col="pid",
            dsname_col="dsname",
            verts_col="vert_label",
            check_complete=True,
            apply_excel_filter=True,
        )

        # read optional pid lists from spec (expect lists of strings)
        pid_fix_1820 = self.spec.get("pid_fix_1820", None)
        pid_fix_28 = self.spec.get("pid_fix_28", None)

        # Compute corrections across the union of train+val files so the same mapping
        # is applied to both datasets (prevents split-dependent differences).
        all_files = list(self.files) + list(val_files)
        pid_corrections_all = self._compute_pid_corrections(
            all_files,
            target_pids_1820=pid_fix_1820,
            target_pids_28=pid_fix_28
        )

        self.label_override_gt_summary = None
        if self.gt_label_override:
            self._apply_gt_label_override(pid_corrections_all)

        # Print only the actual corrections. Previously the first 10 entries were shown;
        # without a filter those are almost always identity mappings, giving the false impression
        # that nothing is being corrected.
        if pid_corrections_all:
            self.pid_correction_summary = print_pid_correction_summary(pid_corrections_all)
        else:
            self.pid_correction_summary = summarize_pid_corrections({})

        # ---- attach the same computed corrections to both dataset instances ----
        self.train_dataset = VerSeDataset(files, transform=self.train_transforms,
                                          label_mode=self.label_mode)
        self.train_dataset.pid_corrections = pid_corrections_all

        self.val_dataset = VerSeDataset(val_files, transform=self.val_transforms,
                                        label_mode=self.label_mode)
        self.val_dataset.pid_corrections = pid_corrections_all

        self._apply_label_overrides()

        print(f"Label space: {self.label_mode} ({self.num_classes} classes)")

    def _apply_label_overrides(self):
        """
        Apply spec["label_overrides"] (CSV id, label) to train and val datasets.

        Every id must appear in train or val — an override file that points nowhere is almost
        always a mistake (wrong run, wrong correction) and should fail loudly rather than
        silently doing nothing. Label REMOVE_LABEL (-1) removes the sample.
        """
        self.label_overrides_summary = None
        src = self.spec.get("label_overrides")
        if not src:
            return
        overrides = load_label_overrides(src)
        known = {}
        for name, ds in (("train", self.train_dataset), ("val", self.val_dataset)):
            for p in ds.file_paths:
                known[ds.sample_id_for(p)] = name
        unknown = sorted(set(overrides) - set(known))
        if unknown:
            raise ValueError(f"{len(unknown)} override ids not in the dataset, e.g. {unknown[:5]}")
        bad = {i: l for i, l in overrides.items()
               if l != REMOVE_LABEL and not (0 <= l < self.num_classes)}
        if bad:
            raise ValueError(f"Override labels outside 0..{self.num_classes - 1}: {list(bad.items())[:5]}")

        summary = {"file": str(src), "n_rows": len(overrides)}
        for name, ds in (("train", self.train_dataset), ("val", self.val_dataset)):
            mine = {i: l for i, l in overrides.items() if known[i] == name}
            remove = {i for i, l in mine.items() if l == REMOVE_LABEL}
            ds.file_paths = [p for p in ds.file_paths if ds.sample_id_for(p) not in remove]
            changed = sum(1 for p in ds.file_paths
                          if ds.sample_id_for(p) in mine
                          and mine[ds.sample_id_for(p)] != ds.base_label_for(p))
            ds.label_overrides = {i: l for i, l in mine.items() if l != REMOVE_LABEL}
            summary[name] = {"relabelled": len(ds.label_overrides), "label_changed": changed,
                             "removed": len(remove), "n_samples": len(ds.file_paths)}
        # self.files is the training file list read by __main__ and notebooks.
        self.files = list(self.train_dataset.file_paths)
        self.label_overrides_summary = summary
        print(f"Label overrides from {src}: {summary}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate,
            # Seeds MONAI augmentations per worker/epoch (see seed_worker). Val and
            # scoring have no random transforms and do not need this.
            worker_init_fn=seed_worker,
        )
    
    def scoring_dataloader(self, split: str = "train"):
        """
        Deterministic loader for post-hoc scoring (AUM, VoG, GLARE), not for training.

        Differences from train_dataloader():
          - shuffle=False: fixed, sorted order (get_data_VerSe sorts the file list).
          - no worker_init_fn: the scoring transform has no random components.

        pid_corrections and label_overrides are copied from the respective dataset so that
        labels and sample_ids are identical to those used in training — the id is the
        join key throughout the pipeline.

        split="val" scores the validation set. The training transform is used, as in GLARE scoring.
        """
        if split not in ("train", "val"):
            raise ValueError(f"split must be train or val, not {split!r}")
        source = self.train_dataset if split == "train" else self.val_dataset

        ds = VerSeDataset(list(source.file_paths), transform=self.train_dataset.transform,
                          label_mode=self.label_mode)
        ds.pid_corrections = source.pid_corrections
        ds.label_overrides = source.label_overrides
        print(f"Scoring {split}: {len(ds.file_paths)} samples")

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate
        )

def check_image_sizes(file_paths):
    sizes = {}
    for path in file_paths:
        data = np.load(path)
        image_3d = data['img']  # shape (D, H, W)
        shape = image_3d.shape
        if shape not in sizes:
            sizes[shape] = 0
        sizes[shape] += 1
    
    print(f"Distinct image sizes and their counts:")
    for shape, count in sizes.items():
        print(f"  Size {shape}: {count} images")
    print(f"Number of distinct sizes: {len(sizes)}")
    return sizes



if __name__ == "__main__":
    spec = {
        "dataset_dir": str(paths.DATASET_DIR),
        "excel_path": str(paths.EXCEL_FILTER),
        "split": "train", 
        "batch_size": 4
    }

    data_module = VerSeDataLoader(spec=spec, num_workers=0)
    data_module.setup()

    # # --- Sanity prints ---
    print("Value range of the first 5 training images:")
    for path in data_module.files[:5]:
        data = np.load(path)
        img = data['img']
        print(f"{path}: min={img.min()}, max={img.max()}")



    # check training set
    #print("Training set:")
    # check image sizes
    #check_image_sizes(data_module.files)

    #train_loader = data_module.train_dataloader()
    #batch_imgs, batch_labels, batch_pids = next(iter(train_loader))
    #print(f"Train batch images shape: {batch_imgs.shape}")
    #print(f"Train batch labels: {batch_labels}")
    #print(f"Train batch pids: {batch_pids}")


    # check validation files
    #print("\nValidation set:")
    #if data_module.val_dataset is not None:
     #   val_files = data_module.val_dataset.file_paths
      #  print(f"Collected {len(val_files)} validation files.")
       # #check_image_sizes(val_files)
        #val_loader = data_module.val_dataloader()
        #val_batch_imgs, val_batch_labels, val_batch_pids = next(iter(val_loader))
        #print(f"Validation batch images shape: {val_batch_imgs.shape}")
        #print(f"Validation batch labels: {val_batch_labels}")
        #print(f"Validation batch pids: {val_batch_pids}")
    #else:
        #print("No validation dataset found!")