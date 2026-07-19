import os
import json
from pytorch_lightning.callbacks import Callback

class StepLossLogger(Callback):
    def __init__(self, log_dir, log_interval=30):
        super().__init__()
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.train_log = []
        self.val_log = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval == 0:
            loss = outputs['loss'].item() if isinstance(outputs, dict) and 'loss' in outputs else outputs.item()
            self.train_log.append({"step": batch_idx, "loss": loss})
            with open(os.path.join(self.log_dir, "train_loss_log.json"), "w") as f:
                json.dump(self.train_log, f, indent=4)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.log_interval == 0:
            loss = outputs['loss'].item() if isinstance(outputs, dict) and 'loss' in outputs else outputs.item()
            self.val_log.append({"step": batch_idx, "loss": loss})
            with open(os.path.join(self.log_dir, "val_loss_log.json"), "w") as f:
                json.dump(self.val_log, f, indent=4)
    
    #def on_train_end(self, trainer, pl_module):
    #    # accuracy ausrechnen inkl richtige argumente eingeben
    #    with open(os.path.join(self.log_dir, "acc_log.json"), "w") as f:
    #        json.dump('accuracy variable', f, indent=4)
    #    return super().on_train_end(trainer, pl_module)