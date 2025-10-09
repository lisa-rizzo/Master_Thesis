"""
Script to launch training with GLARE(x)
"""

from models.model import Model
from dataloaders.CIFAR10DataLoader import CIFAR10DataLoader
from glare import compute_glare
from helper.arguments import get_parameter

from monai.transforms import (
    Compose,
    Resize,
    CenterSpatialCrop,
    NormalizeIntensity,
    ToTensor,
    RandFlip,
    RandAdjustContrast,
    RandAffine
)
import numpy as np
from PIL import Image
from torchvision import transforms
import datetime
import pytz
from pytorch_lightning import Trainer
import os
import json
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef


########## Set specs here ##########

spec = {
    "directory": "glare_clean/results",
    "dataset_dir": "glare_clean/datasets/CIFAR10",
    "job_name": "test_glare",
    #### Model training settings
    "model": "dense169",
    "pretrained": True,
    "noise_level": "rand1",
    "drop_rate": 0.05,
    "epochs": 2,
    "lr": 1e-6,
    "lr_scheduler": True,
    "lr_end_factor": 0.01,
    "batch_size": 64,
    "holdout_set_size": 0.05, # Fraction or 0 if no holdout set performance 
    "use_train_for_val": False,
    "train_set_size": 0.01, # Split between train and validation set (entire set - holdout_set) (only relevant if "use_train_for_val" is False)
    #### GLARE settings
    "iterations": 2,
    "remove_mislabels": True,
    "correct_mislabels": False,
    "method": "glarex",
    "threshold_fraction": 0.1
}

spec_data = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "resize": 256,
    "centerCrop": 224
}
##############################################



########## Set data augmentation here ##########
class ImageTransform():
    def __init__(self, is_training: bool = True, resize: int = 144, crop: int = 128, mean: list = [0.5, 0.5, 0.5], std: list = [0.5, 0.5, 0.5]):

        if is_training:
            self.transforms = Compose([
                ToTensor(),
                NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True), 
                Resize((resize, resize)),
                CenterSpatialCrop((crop, crop)), 
                RandFlip(spatial_axis=1, prob=0.2),  
                RandAdjustContrast(prob=0.2, gamma=(0.8, 1.5)),
                RandAffine(
                    prob=0.2,  
                    rotate_range=(np.radians(10), np.radians(10)),  
                    translate_range=(0.05, 0.05), 
                    mode="bilinear",  
                    padding_mode="zeros" 
                )                 
            ])
        else:
            self.transforms = Compose([
                ToTensor(),
                NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True), 
                Resize((resize, resize)),
                CenterSpatialCrop((crop, crop))
            ])

    def __call__(self, data):
        if isinstance(data, Image.Image):  # Check if it's a PIL image
            data = transforms.ToTensor()(data)
        data = self.transforms(data)
        return data
##############################################


    
def _get_unique_filename(base_name: str = "unnamed"):
        timestamp = datetime.datetime.now(pytz.timezone("Europe/Berlin")).strftime("%d%m%Y_%H%M")
        return f"{base_name}_{timestamp}"
    
    

def main():
    if get_parameter(spec, "remove_mislabels", False, bool) and get_parameter(spec, "correct_mislabels", False, bool):
        raise Exception("Can't remove and correct mislabeles at the same time :/")

    method = get_parameter(spec, "method", "", str)
    if (get_parameter(spec, "remove_mislabels", False, bool) or get_parameter(spec, "correct_mislabels", False, bool)) and (method != "glare" and method != "glarex"):
        raise Exception("unkown method! Please choose either \"glare\" or \"glarex\".")

    job_directory = get_parameter(spec, 'directory', 'unnamed', str) + "/" + _get_unique_filename(get_parameter(spec, "job_name", "unnamed", str))
    os.makedirs(job_directory, exist_ok=True)

    with open(f"{job_directory}/spec.json", "w") as f:
        json.dump(spec, f)


    ########## Set data module here ##########
    data_module = CIFAR10DataLoader(
            spec=spec,
            noise_level=get_parameter(spec, 'noise_level', 'rand1', str),
            train_transforms=ImageTransform(is_training=True, resize=get_parameter(spec_data, "resize", 256, int), crop=get_parameter(spec_data, "centerCrop", 224, int), mean=spec_data["mean"], std=spec_data["std"]),
            val_transforms=ImageTransform(is_training=False, resize=get_parameter(spec_data, "resize", 256, int), crop=get_parameter(spec_data, "centerCrop", 224, int), mean=spec_data["mean"], std=spec_data["std"]),
            num_workers=12
        )

    data_module.setup()
    num_classes = data_module.num_classes

    ##########################################

    if get_parameter(spec, "holdout_set_size", 0.0, float):
        holdout_loader = data_module.holdout_dataloader()

    for i in range(get_parameter(spec, "iterations", 1, int)):
        weights_dir = f"{job_directory}/iteration{i}"
        os.makedirs(weights_dir, exist_ok=True)

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        ########## Set model here ##########
        model = Model(
            num_classes=num_classes,
            spec=spec,
            weights_dir=weights_dir
        )
        ##########################################  

        trainer = Trainer(accelerator='auto', devices='auto', max_epochs=get_parameter(spec, "epochs", 25, int))
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


        # Compute scores
        scores, grad_norms = compute_glare(dataloader=train_loader, num_classes=num_classes, weights_dir=weights_dir, spec=spec)

        if get_parameter(spec, "remove_mislabels", False, bool):
            threshold = scores[f'{method} score'].quantile(get_parameter(spec, "threshold_fraction", 0.1, float))
            mislabels = scores[scores[f'{method} score'] <= threshold]
            scores['removed'] = scores.index.isin(mislabels.index).astype(int)

            mislabels_id = list(mislabels["id"])
            data_module.remove_mislabels(mislabels_id)
            

        if get_parameter(spec, "correct_mislabels", False, bool):
            threshold = scores[f'{method} score'].quantile(get_parameter(spec, "threshold_fraction", 0.1, float))
            mislabels = scores[scores[f'{method} score'] <= threshold]
            scores['corrected'] = scores.index.isin(mislabels.index).astype(int)

            mislabels_id = list(mislabels["id"])
            mislabels_pred_label = list(mislabels[f'alternative class ({method})'])
            data_module.correct_mislabels(mislabels_id, mislabels_pred_label)

        scores.to_pickle(f'{weights_dir}/scores.pkl')
        grad_norms.to_pickle(f'{weights_dir}/grad_norms.pkl')


    if get_parameter(spec, "holdout_set_size", 0.0, float):
        model.eval() 
        all_preds = []
        all_targets = []
        all_losses = []
        CE = nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad(): 
            for batch in holdout_loader:
                images, labels, ids, targets = batch
                outputs = model(images)

                loss = torch.mean(CE(outputs, targets))
                all_losses.append(loss.item())

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.detach().cpu().numpy()) 
                all_targets.extend(targets.detach().cpu().numpy())

        average_loss = np.mean(all_losses)
        accuracy = accuracy_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')
        mcc = matthews_corrcoef(all_targets, all_preds)

        metrics = {
            "avg_loss": average_loss,
            "accuracy": accuracy,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc
        }

        # Save as JSON
        with open(job_directory + "/holdout_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)



if __name__ == '__main__':
    main()