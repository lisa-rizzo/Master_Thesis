from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import torch
import pandas as pd

LABELS = {
    "clean": "clean_label",
    "worse": "worse_label",
    "aggre": "aggre_label",
    "rand1": "random_label1",
    "rand2": "random_label2",
    "rand3": "random_label3"
    }

class CIFAR10Noise(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ids = list(range(len(self.targets)))
        # Default using the correct labels, if not changed later
        self.labels = self.targets

    def set_label(self, noise_level):
        label_file = torch.load(f'{self.root}/CIFAR-10_human.pt', weights_only=False)
        self.labels = list(list(label_file[LABELS[noise_level]]))

    def __getitem__(self, index):
        image, fine_label = super().__getitem__(index)

        label = self.labels[index]
        id = self.ids[index]
        target = self.targets[index]
        return image, label, id, target
    


class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None, indices=None):

        if indices is not None:
            # Create a new list from the dataset for the given indices.
            dataset = [dataset[i] for i in indices]
        self.dataset = dataset
        self.transform = transform

    def remove_samples(self, mislabels_id):
        for id_to_remove in mislabels_id: 
            idx = next((i for i, t in enumerate(self.dataset) if t[2] == id_to_remove), None)
            self.dataset.pop(idx)


    def correct_labels(self, mislabels_id, mislabels_pred_label):
        for id_to_correct, pred_label in zip(mislabels_id, mislabels_pred_label): 
            idx = next((i for i, t in enumerate(self.dataset) if t[2] == id_to_correct), None)
            image, _, id_val, target = self.dataset[idx]
            self.dataset[idx] = (image, pred_label, id_val, target)

    def __getitem__(self, index):
        image, label, id, target = self.dataset[index]

        if self.transform:
            image = self.transform(image)
        return image, label, id, target

    def __len__(self):
        return len(self.dataset)
    
    @property
    def ids(self):
        return [self.dataset[i][2] for i in range(len(self.dataset))]

    @property
    def labels(self):
        return [self.dataset[i][1] for i in range(len(self.dataset))]

    @property
    def targets(self):
        return [self.dataset[i][3] for i in range(len(self.dataset))]
    

