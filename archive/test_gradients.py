from archive.glare_resnet import compute_glare
from glare_clean.dataloaders.dataloader_VerSe import VerSeDataLoader
import torch

# Spezifikation wie im Training
spec = {
    "dataset_dir": "/DATA/anjany/prepared",
    "excel_path": "/home/student/lisa_ma/data_filter_joined.xlsx",
    "batch_size": 2  # Nur kleiner Batch!
}
num_classes = 4
weights_dir = "/home/student/lisa_ma/glare_clean/glare_clean/results/test_glare_16092025_2258/weights"

# Dataloader initialisieren
data_module = VerSeDataLoader(spec=spec, num_workers=0)
data_module.setup()
train_dataloader = data_module.train_dataloader()

# Nur einen Batch nehmen
small_batch = next(iter(train_dataloader))

# Modell laden (nur erstes Weight für Test)
import os
from models.model import Model
import re

weight_files = [
    os.path.join(weights_dir, f)
    for f in os.listdir(weights_dir)
    if re.match(r'epoch_\d+$', f)
]
weight_files.sort(key=lambda x: int(x.rsplit('_', 1)[-1]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_files = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.startswith("epoch_")]
weight_files.sort(key=lambda x: int(x.rsplit('_', 1)[-1]))
first_weight = weight_files[0]

model = Model(num_classes=num_classes, spec=spec, weights_dir=weights_dir).model.to(device)
state_dict = torch.load(first_weight, map_location=device)
new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# Gradienten berechnen
inputs, labels, ids = small_batch
inputs, labels = inputs.to(device), labels.to(device)
inputs.requires_grad = True

import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
outputs = model(inputs)
loss = loss_fn(outputs, labels)
loss.backward()

for name, param in model.named_parameters():
    print(f"{name} grad:", param.grad)

print("Gradientenberechnung für kleinen Batch erfolgreich ausgeführt.")