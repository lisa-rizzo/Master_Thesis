import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import F1Score, Accuracy, MatthewsCorrCoef
import json
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
from helper.arguments import get_parameter


def get_model(id: str, num_classes: int, pretrained: bool):
    """
    Gibt ein 3D-CNN-Modell zurück, angepasst für medizinische 3D-Graustufendaten.

    Parameter:
    - id: Modellname, z.B. "r3d_18", "mc3_18", "r2plus1d_18"
    - num_classes: Anzahl Ausgangsklassen
    - pretrained: bool, ob vortrainierte Gewichte geladen werden sollen
    """

    pretrained_flag = pretrained if isinstance(pretrained, bool) else False

    if id == "r3d_18":
        model = r3d_18(pretrained=pretrained_flag)
    elif id == "mc3_18":
        model = mc3_18(pretrained=pretrained_flag)
    elif id == "r2plus1d_18":
        model = r2plus1d_18(pretrained=pretrained_flag)
    else:
        raise ValueError(f"Unbekanntes 3D Modell: {id}")

    # Erste Conv3d Schicht: statt 3 Kanäle nur 1 Kanal (Graustufen)
    conv1 = model.stem[0]
    model.stem[0] = nn.Conv3d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=(conv1.bias is not None)
    )

    # Letzte Full-Connected-Schicht anpassen
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class Model(pl.LightningModule):
    def __init__(self, num_classes: int, spec: dict, weights_dir: str):
        super().__init__()
        
        self.num_classes = num_classes
        self.spec = spec
        self.id = get_parameter(spec, "model", "r3d_18", str)   # Standardmäßig 3D ResNet18
        self.pretrained = get_parameter(spec, "pretrained", True, bool)
        self.model = get_model(id=self.id, num_classes=self.num_classes, pretrained=self.pretrained)
        self.epoch_metrics = []

        self.softmax = nn.Softmax(dim=1)
        self.CE = nn.CrossEntropyLoss(reduction="none")
        self.lr = get_parameter(spec, "lr", 1e-6, float)
        self.use_lr_scheduler = get_parameter(spec, "lr_scheduler", True, bool)
        self.lr_end_factor = get_parameter(spec, "lr_end_factor", 0.001, float)
        self.epochs = get_parameter(spec, "epochs", 25, int)

        self.weight_path = weights_dir
        self.lr_list = []

        self.validation_losses = []
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro') 
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        self.mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters() #

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, labels, ids = batch
        
        loss, logits, logits_softmax = self._shared_step(X, labels)
        preds = torch.argmax(logits, dim=1)

        f1 = self.f1_score(preds, labels)
        acc = self.accuracy(preds, labels)
        mcc = self.mcc(preds, labels)

        self.log("train_loss", loss.detach().cpu(), prog_bar=False, on_epoch=True)
        self.log("lr", self._get_current_lr(), prog_bar=True, on_epoch=True)
        self.log("train_f1", f1.detach().cpu(), prog_bar=False, on_epoch=True)
        self.log("train_accuracy", acc.detach().cpu(), prog_bar=False, on_epoch=True)
        self.log("train_mcc", mcc.detach().cpu(), prog_bar=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self, *args, **kwargs):
        os.makedirs(self.weight_path, exist_ok=True)
        torch.save(self.state_dict(), f'{self.weight_path}/epoch_{self.current_epoch}')
        self.lr_list.append(self._get_current_lr())

    def on_train_end(self):
        with open(f'{self.weight_path}/lr.json', "w") as f:
            json.dump(self.lr_list, f)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        X, labels, ids = batch
        loss, logits, logits_softmax = self._shared_step(X, labels)
        preds = torch.argmax(logits, dim=1)

        f1 = self.f1_score(preds, labels)
        acc = self.accuracy(preds, labels)
        mcc = self.mcc(preds, labels)

        self.log("val_loss", loss.detach().cpu(), prog_bar=True, on_epoch=True)
        self.log("val_f1", f1.detach().cpu(), prog_bar=True, on_epoch=True)
        self.log("val_accuracy", acc.detach().cpu(), prog_bar=True, on_epoch=True)
        self.log("val_mcc", mcc.detach().cpu(), prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        print(f"Using lr: {self.lr}")
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.use_lr_scheduler:
            print("Init lr scheduler")
            lr_scheduler = {
                'scheduler': optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=self.lr_end_factor, total_iters=self.epochs),  
                'interval': 'epoch'
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

    def _shared_step(self, X, labels, detach2cpu: bool = False):
        logits = self(X)
        logits_softmax = self.softmax(logits)
        loss = torch.mean(self.CE(logits, labels))
        return loss, logits, logits_softmax

    def _get_current_lr(self):
        optimizer = self.trainer.optimizers[0]
        return optimizer.param_groups[0]['lr']

    def on_validation_epoch_end(self):
        metrics = {
            "epoch": self.current_epoch,
            "val_loss": self.trainer.callback_metrics.get("val_loss", None),
            "val_f1": self.trainer.callback_metrics.get("val_f1", None),
            "val_accuracy": self.trainer.callback_metrics.get("val_accuracy", None),
            "val_mcc": self.trainer.callback_metrics.get("val_mcc", None),
            "train_loss": self.trainer.callback_metrics.get("train_loss", None),
            "train_f1": self.trainer.callback_metrics.get("train_f1", None),
            "train_accuracy": self.trainer.callback_metrics.get("train_accuracy", None),
            "train_mcc": self.trainer.callback_metrics.get("train_mcc", None),
        }
        for k, v in metrics.items():
            if hasattr(v, "item"):
                metrics[k] = float(v.item())
        self.epoch_metrics.append(metrics)
        out_path = os.path.join(self.weight_path, "epoch_metrics.json")
        with open(out_path, "w") as f:
            json.dump(self.epoch_metrics, f, indent=4)
