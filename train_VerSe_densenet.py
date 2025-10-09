"""
Script to launch training with MONAI DenseNet for 3D mediclass ImageTransform:
    def __init__(self, mean, std):
        self.transforms = Compose([
            NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True),
          #  RandSimulateLowResolutiond(keys=["img"], zoom_range=(0.5, 1.0), prob=1.0),
         #   RandAdjustContrastd(keys=["img"], prob=0.5, gamma=(0.7, 1.5)),
         #   RandGaussiannoised(keys=["img"], prob=0.5, mean=0.0, std=0.1),
        #    RandGaussianSmoothd(keys=["img"], prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)), #Weichzeichnung
         #   RandscaleIntensityd(keys=["img"], factors=0.1, prob=0.5),
          #  RandShiftIntensityd(keys=["img"], offsets=0.1, prob=0.5)
        ])
"""

import os
import json
import datetime
import pytz
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef
from models.model_densenet import DenseNetModel
from pytorch_lightning import Trainer
from dataloaders.dataloader_VerSe import VerSeDataLoader
from monai.transforms import Compose, NormalizeIntensity, RandAdjustContrastd, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd, RandShiftIntensityd, RandSimulateLowResolutiond
from metric_cal import StepLossLogger
from pytorch_lightning.loggers import TensorBoardLogger


########## Set specs here ##########

spec = {
    "directory": "results/VerSe_classifier",
    "dataset_dir": "/home/student/lisa_ma/prepared",  
    "job_name": "test_glare", #glarex
    #### Model training settings
    "model": "densenet169",  # DenseNet variant
    "pretrained": False,  # MONAI DenseNet doesn't have pretrained weights for 3D medical
    "spatial_dims": 3,  # 3D images
    "in_channels": 1,  # Grayscale images
    "init_features": 64,
    "growth_rate": 32,
    "block_config": [6, 12, 32, 32],  # DenseNet-169 configuration
    "dropout_prob": 0.2,
    "noise_level": "rand1",
    "drop_rate": 0.05,
    "epochs": 15,
    "lr": 1e-4,
    "lr_scheduler": True,
    "lr_end_factor": 0.01,
    "batch_size": 4,  
    "holdout_set_size": 0, # Kein Holdout-Set
    "use_train_for_val": False,
    "train_set_size": 0.8, # Split between train and validation set (entire set - holdout_set) (only relevant if "use_train_for_val" is False)
    #### GLARE settings
    "iterations": 2, # was 2
    "remove_mislabels": True,
    "correct_mislabels": False,
    "method": "glarex",
    "threshold_fraction": 0.1
}

spec_data = {
    "mean": [-111.3885],
    "std": [406.3665]
}

########## Set data augmentation here ##########

class ImageTransform:
    def __init__(self, mean, std):
        self.transforms = Compose([
            NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True),
            # Aktivierte Augmentierungen für bessere Generalisierung
            #RandAdjustContrastd(keys=["img"], prob=0.3, gamma=(0.8, 1.2)),
            #RandGaussianNoised(keys=["img"], prob=0.2, mean=0.0, std=0.05),
           # RandScaleIntensityd(keys=["img"], factors=0.05, prob=0.3),
            #RandShiftIntensityd(keys=["img"], offsets=0.05, prob=0.3)
        ])

    def __call__(self, data):
        return self.transforms(data)

##########################################

def _get_unique_filename(base_name: str = "unnamed"):
        timestamp = datetime.datetime.now(pytz.timezone("Europe/Berlin")).strftime("%d%m%Y_%H%M")
        return f"{base_name}_{timestamp}"

def get_parameter(spec, key, default, typ):
    return typ(spec[key]) if key in spec else default

# Results directory 
job_directory = get_parameter(spec, 'directory', 'unnamed', str) + "/" + _get_unique_filename(get_parameter(spec, "job_name", "unnamed", str))
os.makedirs(job_directory, exist_ok=True)

# safe specifications
with open(f"{job_directory}/spec.json", "w") as f:
    json.dump(spec, f, indent=4)

########## Set data module ##########
data_module = VerSeDataLoader(
        spec=spec,
        train_transforms=ImageTransform(mean=spec_data["mean"], std=spec_data["std"]),
        val_transforms=ImageTransform(mean=spec_data["mean"], std=spec_data["std"]),
        num_workers=6  # Reduziert wegen Speicherproblemen
    )

data_module.setup()
num_classes = 4

##########################################

weights_dir = f"{job_directory}/weights"
os.makedirs(weights_dir, exist_ok=True)

########## Set model here ##########
model = DenseNetModel(
    num_classes=num_classes, 
    spec=spec,
    weights_dir=weights_dir
    )
##########################################

loss_logger = StepLossLogger(log_dir=job_directory, log_interval=30)
tb_logger = TensorBoardLogger(save_dir=os.path.dirname(job_directory), name=job_directory.split("/")[-1])

trainer = Trainer(
    accelerator='auto',
    devices='auto',
    max_epochs=get_parameter(spec, "epochs", 25, int),
    callbacks=[loss_logger],  
    logger=tb_logger,         
    check_val_every_n_epoch=1  
)
trainer.fit(model=model, datamodule=data_module)