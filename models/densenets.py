import torch.nn as nn
from torchvision import models


# List of possible DenseNets
def dense121(num_classes, pretrained, drop_rate: float = 0.0):
    network = models.densenet121(weights=pretrained, drop_rate=drop_rate)
    network = setup_densenet_layer(network, num_classes)
    return network


def dense161(num_classes, pretrained, drop_rate: float = 0.0):
    network = models.densenet161(weights=pretrained, drop_rate=drop_rate)
    network = setup_densenet_layer(network, num_classes)
    return network

def dense169(num_classes, pretrained, drop_rate: float = 0.0):
    network = models.densenet169(weights=pretrained, drop_rate=drop_rate)
    network = setup_densenet_layer(network, num_classes)
    return network


def dense201(num_classes, pretrained, drop_rate: float = 0.0):
    network = models.densenet201(weights=pretrained, drop_rate=drop_rate)
    network = setup_densenet_layer(network, num_classes)
    return network



# Adjust DenseNet for the classification task
def setup_densenet_layer(network, num_classes):
    last_in = network.classifier.in_features
    # Replace last layer with new layer that have num_classes nodes
    network.classifier = nn.Linear(last_in, num_classes)
    #network.classifier = nn.Sequential(nn.Linear(last_in, num_classes), nn.Sigmoid())
    return network