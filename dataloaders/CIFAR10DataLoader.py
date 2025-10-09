import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from helper.arguments import get_parameter
#imports custom dataset classes with label noise
from datasets.CIFAR10.CIFAR10_dataset import CIFAR10Noise, CIFAR10Dataset
import torch

class CIFAR10DataLoader(pl.LightningDataModule):
    def __init__(self, spec: dict, noise_level: str, train_transforms, val_transforms, num_workers: int = 4):
        super().__init__()

        
        self.root = get_parameter(spec, "dataset_dir", "datasets/CIFAR10", str) #folder of the dataset 
        self.spec = spec
        self.batch_size = get_parameter(spec, "batch_size", 32, int)
        self.num_workers = num_workers 

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.num_classes = None
        self.use_train_for_val = get_parameter(spec, "use_train_for_val", False, bool)
        self.holdout_set_size = get_parameter(spec, "holdout_set_size", 0.0, float)
        self.train_set_size = get_parameter(spec, "train_set_size", 0.0, float)
        self.noise_level = noise_level


    #prepares dataset for training/validation/holdout
    def setup(self, stage=None):

        #load dataset
        train_dataset = CIFAR10Noise(
            root=self.root,
            train=True,
            download=True
        ) 
        train_dataset.set_label(noise_level=self.noise_level)

        self.num_classes = len(train_dataset.classes)

        self.classes = train_dataset.classes

        self.targets = train_dataset.targets
        self.noisy_targets = train_dataset.labels

        if self.holdout_set_size:
            holdout_size = int(self.holdout_set_size * len(train_dataset)) 
            remaining_size = len(train_dataset) - holdout_size
            holdout_dataset, remaining_dataset = random_split(train_dataset, [holdout_size, remaining_size], generator=torch.Generator().manual_seed(42)) 
        else:
            remaining_dataset = train_dataset
        
        if hasattr(remaining_dataset, 'indices'):
            allowed_indices = remaining_dataset.indices
        else:
            allowed_indices = list(range(len(remaining_dataset)))

        if self.use_train_for_val:
            self.train_dataset = CIFAR10Dataset(
                remaining_dataset.dataset if hasattr(remaining_dataset, 'dataset') else remaining_dataset,
                transform=self.train_transforms,
                indices=allowed_indices  # Assumes CIFAR10Dataset can filter by indices
            )
            self.val_dataset = CIFAR10Dataset(
                remaining_dataset.dataset if hasattr(remaining_dataset, 'dataset') else remaining_dataset,
                transform=self.val_transforms,
                indices=allowed_indices
            )
        else:
            train_size = int(self.train_set_size * len(allowed_indices))
            train_indices = allowed_indices[:train_size]
            val_indices = allowed_indices[train_size:]
            self.train_dataset = CIFAR10Dataset(
                remaining_dataset.dataset if hasattr(remaining_dataset, 'dataset') else remaining_dataset,
                transform=self.train_transforms,
                indices=train_indices
            )
            self.val_dataset = CIFAR10Dataset(
                remaining_dataset.dataset if hasattr(remaining_dataset, 'dataset') else remaining_dataset,
                transform=self.val_transforms,
                indices=val_indices
            )

        if self.holdout_set_size:
            self.holdout_dataset = CIFAR10Dataset(
                holdout_dataset.dataset if hasattr(holdout_dataset, 'dataset') else holdout_dataset,
                transform=self.val_transforms,
                indices=holdout_dataset.indices
        )


    #removes mislabeled samples from trainig set (and val if they overlap)
    def remove_mislabels(self, mislabels_id):
        self.train_dataset.remove_samples(mislabels_id)
        
        if self.use_train_for_val:
            self.val_dataset.remove_samples(mislabels_id)

    #corrects labels on a subset of samples 
    def correct_mislabels(self, mislabels_id, mislabels_pred_label):
        self.train_dataset.correct_labels(mislabels_id, mislabels_pred_label)

        if self.use_train_for_val:
            self.val_dataset.remove_samples(mislabels_id)

        

    #dataloader methods return PyTorch dataloader for each split
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn=worker_init_fn)
    
    def holdout_dataloader(self):
        if self.holdout_set_size:
            return DataLoader(self.holdout_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn=worker_init_fn)
        else:
            return None
    

#ensures each dataloader worker has a unique, reproducible random seed
def worker_init_fn(worker_id):
    import numpy as np
    import random
    import torch

    base_seed = torch.initial_seed()

    # Ensure seed stays within 32-bit range 
    worker_seed = base_seed + worker_id
    if worker_seed >= 2**30: 
        worker_seed = worker_seed % (2**30)

    random.seed(worker_seed)
    np.random.seed(worker_seed)