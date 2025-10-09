import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import numpy as np
import datasets.VerSe.get_data_VerSe as get_data_VerSe  
from pathlib import Path
import re
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F


class VerSeDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        file_paths: list of Path objects to .npz files containing 3D grayscale images
        transform: optional transform to apply on the 3D image numpy array (e.g. normalization, augmentation)
        """
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path)
        image_3d = data['img']  # shape (D, H, W), grayscale
        image_tensor = torch.tensor(image_3d, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        label = self.get_region_label_from_vert(self.get_vert_number(path))
        pid = self.get_pid_from_path(path)
        return image_tensor, label, pid
    
    @staticmethod
    def get_vert_number(path):
        path_str = str(path)
        match = re.search(r'_vert(\d+)', path_str)
        if match:
            return int(match.group(1))
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

        self.train_dataset = VerSeDataset(files, transform=self.train_transforms)

        # Val files
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
        self.val_dataset = VerSeDataset(val_files, transform=self.val_transforms)

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

    # --- Hier Wertebereich der ersten 5 Trainingsbilder ausgeben ---
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