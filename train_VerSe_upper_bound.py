"""
Upper-bound training: same config as label_override_8ep but with all 109 GT mislabels
corrected via label_overrides. Used as the theoretical ceiling for GLARE evaluation.
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
from pytorch_lightning import Trainer, seed_everything
from dataloaders.dataloader_VerSe_new import VerSeDataLoader
from monai.transforms import Compose, NormalizeIntensityd, RandAdjustContrastd, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd, RandShiftIntensityd, RandSimulateLowResolutiond
from helper.loss_logger import StepLossLogger
from pytorch_lightning.loggers import TensorBoardLogger

# C1: reproducibility — fixes weight init, data ordering, and augmentation RNG draws
seed_everything(42, workers=True)
# from monai.utils import set_determinism; set_determinism(seed=42)  # uncomment for full MonAI determinism


########## Set specs here ##########

spec = {
    "directory": "results/VerSe_upper_bound",
    "dataset_dir": "/home/student/lisa_ma/prepared",
    "job_name": "upper_bound_8ep",
    #### Model training settings
    "model": "densenet169",
    "pretrained": False,
    "spatial_dims": 3,
    "in_channels": 1,
    "init_features": 64,
    "growth_rate": 32,
    "block_config": [6, 12, 32, 32],
    "dropout_prob": 0.2,
    "noise_level": "rand1",
    "drop_rate": 0.05,
    "epochs": 8,
    "lr": 1e-4,
    "lr_scheduler": True,
    "lr_end_factor": 0.01,
    "batch_size": 4,
    "holdout_set_size": 0,
    "use_train_for_val": False,
    "train_set_size": 0.8,
    #### Data augmentation (identical to baseline run)
    "augmentations": {
        "enabled": True,
        "rand_adjust_contrast": {"prob": 0.1, "gamma": [0.8, 2.0]},
        "rand_gaussian_noise":  {"prob": 0.1, "mean": 0.0, "std": 0.05},
        "rand_scale_intensity": {"prob": 0.1, "factors": [-0.05, 0.05]},
        "rand_shift_intensity": {"prob": 0.1, "offsets": [-0.05, 0.05]},
    },
    #### Upper bound: apply all 109 GT mislabel corrections to train labels
    "gt_corrections_path": "evaluation/gt_mislabels_complete_LO.xlsx",
    #### GLARE settings (no iterative GLARE loop for upper bound)
    "iterations": 2,
    "remove_mislabels": False,
    "correct_mislabels": False,
    "method": "glarex",
    "threshold_fraction": 0.1
}
pid_json_path = "/home/student/lisa_ma/datasets/VerSe/pid_corrections.json"
if os.path.isfile(pid_json_path):
    try:
        with open(pid_json_path, "r") as f:
            pid_data = json.load(f)
        if pid_data.get("pid_fix_1820"):
            spec["pid_fix_1820"] = pid_data["pid_fix_1820"]
        if pid_data.get("pid_fix_28"):
            spec["pid_fix_28"] = pid_data["pid_fix_28"]
        print("Loaded pid corrections into spec:", [k for k in ("pid_fix_1820","pid_fix_28") if k in spec])
    except Exception as e:
        print("Warning: could not load pid_corrections.json:", e)
else:
    print("pid_corrections.json not found, continuing without pid fixes.")

spec_data = {
    "mean": [-111.3885],
    "std": [406.3665]
}

########## Set data augmentation here ##########

class ImageTransform:
    def __init__(self, mean, std, aug_cfg=None):
        transforms = [
            NormalizeIntensityd(keys=["img"], subtrahend=mean, divisor=std, channel_wise=True),
        ]
        if aug_cfg and aug_cfg.get("enabled", False):
            if "rand_adjust_contrast" in aug_cfg:
                c = aug_cfg["rand_adjust_contrast"]
                transforms.append(RandAdjustContrastd(keys=["img"], prob=c["prob"], gamma=tuple(c["gamma"])))
            if "rand_gaussian_noise" in aug_cfg:
                c = aug_cfg["rand_gaussian_noise"]
                transforms.append(RandGaussianNoised(keys=["img"], prob=c["prob"], mean=c["mean"], std=c["std"]))
            if "rand_scale_intensity" in aug_cfg:
                c = aug_cfg["rand_scale_intensity"]
                transforms.append(RandScaleIntensityd(keys=["img"], factors=tuple(c["factors"]), prob=c["prob"]))
            if "rand_shift_intensity" in aug_cfg:
                c = aug_cfg["rand_shift_intensity"]
                transforms.append(RandShiftIntensityd(keys=["img"], offsets=tuple(c["offsets"]), prob=c["prob"]))
        self.transforms = Compose(transforms)

    def __call__(self, data):
        if isinstance(data, dict):
            out = self.transforms(data)
            return out["img"]
        else:
            out = self.transforms({"img": data})
            return out["img"]

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
        train_transforms=ImageTransform(mean=spec_data["mean"], std=spec_data["std"], aug_cfg=spec.get("augmentations")),
        val_transforms=ImageTransform(mean=spec_data["mean"], std=spec_data["std"], aug_cfg=None),
        num_workers=6
    )

data_module.setup()
print("Applied pid_corrections (train):", len(getattr(data_module.train_dataset, "pid_corrections", {})))
for i, (pid, mapping) in enumerate(data_module.train_dataset.pid_corrections.items()):
    if i >= 10:
        break
    print(pid, mapping)
print("GT label corrections active (train):", len(getattr(data_module.train_dataset, "gt_label_corrections", {})))
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
