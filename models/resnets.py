import torch.nn as nn
from torchvision import models


# List of possible ResNets
def resnet18(num_classes, pretrained, drop_rate=None):
    network = models.resnet18(weights=pretrained)  # (weights=models.ResNet18_Weights.DEFAULT)
    network = setup_resnet_layer(network, num_classes)
    return network


def resnet34(num_classes, pretrained, drop_rate=None):
    network = models.resnet34(weights=pretrained)  # (weights=models.ResNet34_Weights.DEFAULT)
    network = setup_resnet_layer(network, num_classes)
    return network


def resnet50(num_classes, pretrained, drop_rate=None):
    network = models.resnet50(weights=pretrained)  # (weights=models.ResNet50_Weights.DEFAULT)
    network = setup_resnet_layer(network, num_classes)
    return network


def resnet101(num_classes, pretrained, drop_rate=None):
    network = models.resnet101(weights=pretrained)  # (weights=models.ResNet101_Weights.DEFAULT)
    network = setup_resnet_layer(network, num_classes)
    return network


def resnet152(num_classes, pretrained, drop_rate=None):
    network = models.resnet152(weights=pretrained)  # (weights=models.ResNet152_Weights.DEFAULT)
    network = setup_resnet_layer(network, num_classes)
    return network



# Adjust ResNet for the classification task
def setup_resnet_layer(network, num_classes):
    last_in = network.fc.in_features
    # Replace last layer with new layer that have num_classes nodes
    network.fc = nn.Linear(last_in, num_classes)
    #network.fc = nn.Sequential(nn.Linear(last_in, num_classes), nn.Sigmoid())
    return network
