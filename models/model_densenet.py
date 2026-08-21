import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import F1Score, Accuracy, MatthewsCorrCoef
import json
from monai.networks.nets import DenseNet
from helper.arguments import get_parameter


class DenseNetModel(pl.LightningModule):
    def __init__(self, num_classes: int, spec: dict, weights_dir: str):
        super().__init__()
        
        self.num_classes = num_classes
        self.spec = spec
        self.spatial_dims = get_parameter(spec, "spatial_dims", 3, int)
        self.in_channels = get_parameter(spec, "in_channels", 1, int)
        self.init_features = get_parameter(spec, "init_features", 64, int)
        self.growth_rate = get_parameter(spec, "growth_rate", 32, int)
        self.block_config = get_parameter(spec, "block_config", [6, 12, 24, 16], list)
        self.dropout_prob = get_parameter(spec, "dropout_prob", 0.2, float)
        
        # MONAI DenseNet erstellen
        self.model = DenseNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=num_classes,
            init_features=self.init_features,
            growth_rate=self.growth_rate,
            block_config=tuple(self.block_config),  # Muss Tuple sein
            dropout_prob=self.dropout_prob,
            act=('relu', {'inplace': True}),
            norm='batch'
        )
        
        self.epoch_metrics = []
        self.softmax = nn.Softmax(dim=1)
        self.CE = nn.CrossEntropyLoss(reduction="none")
        self.lr = get_parameter(spec, "lr", 1e-4, float)
        self.use_lr_scheduler = get_parameter(spec, "lr_scheduler", True, bool)
        self.lr_end_factor = get_parameter(spec, "lr_end_factor", 0.001, float)
        self.epochs = get_parameter(spec, "epochs", 25, int)

        self.weight_path = weights_dir
        self.lr_list = []

        self.validation_losses = []
        # B1: separate metric instances for train vs val (torchmetrics objects are stateful;
        # sharing one set across phases mixes their accumulation).
        self.train_f1  = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1    = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        self.train_mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
        self.val_mcc   = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, labels, ids = batch
        
        loss, logits, logits_softmax = self._shared_step(X, labels)
        preds = torch.argmax(logits, dim=1)

        # B2: accumulate over the whole epoch — log the metric OBJECT so Lightning
        # computes the true epoch-level macro metric and resets it (not a noisy per-batch value).
        self.train_f1.update(preds, labels)
        self.train_acc.update(preds, labels)
        self.train_mcc.update(preds, labels)

        self.log("train_loss", loss.detach().cpu(), prog_bar=False, on_step=False, on_epoch=True)
        self.log("lr", self._get_current_lr(), prog_bar=True, on_epoch=True)
        self.log("train_f1", self.train_f1, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_accuracy", self.train_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_mcc", self.train_mcc, prog_bar=False, on_step=False, on_epoch=True)
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

        # B1/B2: use the val-only metric objects, accumulated over the epoch.
        self.val_f1.update(preds, labels)
        self.val_acc.update(preds, labels)
        self.val_mcc.update(preds, labels)

        self.log("val_loss", loss.detach().cpu(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mcc", self.val_mcc, prog_bar=True, on_step=False, on_epoch=True)
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


def get_densenet_model(variant: str, spatial_dims: int, in_channels: int, out_channels: int, **kwargs):
    """
    Hilfsfunktion um verschiedene DenseNet-Varianten zu erstellen
    
    Parameters:
    - variant: "densenet121", "densenet169", "densenet201", "densenet264"
    - spatial_dims: 3 für 3D-Daten
    - in_channels: 1 für Graustufenbilder
    - out_channels: Anzahl Klassen
    """
    
    # DenseNet Konfigurationen
    configs = {
        "densenet121": {"block_config": (6, 12, 24, 16), "init_features": 64, "growth_rate": 32},
        "densenet169": {"block_config": (6, 12, 32, 32), "init_features": 64, "growth_rate": 32},
        "densenet201": {"block_config": (6, 12, 48, 32), "init_features": 64, "growth_rate": 32},
        "densenet264": {"block_config": (6, 12, 64, 48), "init_features": 64, "growth_rate": 32}
    }
    
    if variant not in configs:
        raise ValueError(f"Unbekannte DenseNet-Variante: {variant}. Verfügbar: {list(configs.keys())}")
    
    config = configs[variant]
    config.update(kwargs)  # Override mit user-spezifischen Parametern
    
    return DenseNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        **config
    )