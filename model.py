import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import matplotlib.pyplot as plt
import copy

class CNNClassif(nn.Module):
    """Convolutional neural network classifier for Braille letter images"""
    def __init__(self, num_channels1=16, num_channels2=32, num_channels3=64, num_classes=10):
        super(CNNClassif, self).__init__()
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(3, num_channels1, kernel_size=5, padding=2), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.cnn_layer2 = nn.Sequential(nn.Conv2d(num_channels1, num_channels2, kernel_size=5, padding=2), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.cnn_layer3 = nn.Sequential(nn.Conv2d(num_channels2, num_channels3, kernel_size=3, padding=2), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.linear_layer1 = nn.Sequential(nn.Linear(num_channels3*4*4, 64), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(64, num_classes), nn.ReLU())
        
    def forward(self, x):
        w = self.cnn_layer1(x)
        y = self.cnn_layer2(w)
        z = self.cnn_layer3(y)
        #print(z.shape) # This shape will help you give correct input shape to linear_layer1
        z2 = z.reshape(z.shape[0], -1)
        lin1 = self.linear_layer1(z2)
        out = self.linear_layer2(lin1)
        return out 

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)
    return