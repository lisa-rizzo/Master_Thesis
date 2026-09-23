import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import numpy as np
import datasets.VerSe.get_data_VerSe as get_data_VerSe 
from pathlib import Path
import re
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
from collections import defaultdict


class VerSeDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        file_paths: list of Path objects to .npz files containing 3D grayscale images
        transform: optional transform to apply on the 3D image numpy array (e.g. normalization, augmentation)
        """
        self.file_paths = file_paths
        self.transform = transform
        self.pid_corrections = {}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path)
        image_3d = data['img']  # shape (D, H, W), grayscale
        image_tensor = torch.tensor(image_3d, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        orig_vert = self.get_vert_number(path)
        pid = self.get_pid_from_path(path)

        # apply pid-specific correction if exists
        corrected_vert = orig_vert
        if pid in self.pid_corrections:
            corrected_vert = self.pid_corrections[pid].get(orig_vert, orig_vert)

        label = self.get_region_label_from_vert(corrected_vert)
        sample_id = f"{pid}_vert{corrected_vert}"
        return image_tensor, label, sample_id
    
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

class VerSeDataLoader(pl.LightningDataModule):
    def __init__(self, spec: dict, train_transforms=None, val_transforms=None, num_workers=8):
        super().__init__()
        self.spec = spec
        self.root_dir = Path(spec.get("dataset_dir", "/home/student/lisa_ma/prepared"))
        self.excel_file = spec.get("excel_path", "/home/student/lisa_ma/datasets/VerSe/data_filter_joined.xlsx")
        #self.split = spec.get("split", "train")
        self.batch_size = spec.get("batch_size", 1)
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.train_dataset = None
        self.val_dataset = None

    
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
            extra_sacral_verts=frozenset(),  # extra S2-S4 disabled for this experiment
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
            extra_sacral_verts=frozenset(),  # extra S2-S4 disabled for this experiment
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

        # show a short sample of corrections for debugging (optional)
        if pid_corrections_all:
            print("PID corrections (sample):")
            printed = 0
            for pid, mapping in pid_corrections_all.items():
                print(f"  {pid}: {mapping}")
                printed += 1
                if printed >= 10:
                    break

        # ---- attach the same computed corrections to both dataset instances ----
        self.train_dataset = VerSeDataset(files, transform=self.train_transforms)
        self.train_dataset.pid_corrections = pid_corrections_all

        self.val_dataset = VerSeDataset(val_files, transform=self.val_transforms)
        self.val_dataset.pid_corrections = pid_corrections_all

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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
    
    print(f"Gefundene unterschiedliche Bildgrößen und deren Anzahl:")
    for shape, count in sizes.items():
        print(f"  Größe {shape}: {count} Bilder")
    print(f"Anzahl unterschiedlicher Größen: {len(sizes)}")
    return sizes



if __name__ == "__main__":
    spec = {
        "dataset_dir": "/DATA/anjany/prepared",
        "excel_path": "/home/student/lisa_ma/data_filter_joined.xlsx",
        "split": "train", 
        "batch_size": 4
    }

    data_module = VerSeDataLoader(spec=spec, num_workers=0)
    data_module.setup()

    # # --- Sanity prints ---
    print("Wertebereich der ersten 5 Trainingsbilder:")
    for path in data_module.files[:5]:
        data = np.load(path)
        img = data['img']
        print(f"{path}: min={img.min()}, max={img.max()}")
    # --- Ende Wertebereich-Ausgabe ---


    # Trainings set prüfen
    #print("Training set:")
    # Bildgrößen prüfen
    #check_image_sizes(data_module.files)

    #train_loader = data_module.train_dataloader()
    #batch_imgs, batch_labels, batch_pids = next(iter(train_loader))
    #print(f"Train batch images shape: {batch_imgs.shape}")
    #print(f"Train batch labels: {batch_labels}")
    #print(f"Train batch pids: {batch_pids}")


    # Validierungsfiles prüfen
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
        #print("Kein Validierungs-Dataset vorhanden!")