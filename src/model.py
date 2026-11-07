# src/model.py
import torch.nn as nn
import torch.nn.functional as F
import torch

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 20), nn.Sigmoid(),
            nn.Linear(20, 20),  nn.ReLU(),
            nn.Linear(20, 20),  nn.ReLU(),
            nn.Linear(20, 20),  nn.ReLU(),
            nn.Linear(20, 10)
        )

    def forward(self, x):
        return self.net(x)
    

class MNIST_MLP(nn.Module):
    def __init__(self, input_size=784, hidden_1=128, hidden_2=32, num_classes=10):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_1) # 784 -> 128
        self.fc2 = nn.Linear(hidden_1, hidden_2) # 128 -> 32
        self.fc3 = nn.Linear(hidden_2, num_classes) # 32 -> 10
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image: (Batch, 1, 28, 28) -> (Batch, 784)
        x = F.relu(self.fc1(x))  
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  
        x = self.dropout(x)
        x = self.fc3(x)          
        return x
    
# 1. FCN (Fully Convolutional Network) - 3.9M Parameters
class FCN(nn.Module):
    """
    Fully-Convolutional-Network (FCN)
    Target: ~3.9M parameters for CIFAR-10 (32x32 input).
    """
    def __init__(self, num_classes=10):
        super(FCN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 128x16x16
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 128x16x16 -> 256x8x8 (~0.3M params)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 256x8x8 -> 512x4x4 (~1.1M params)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: 512x4x4 -> 512x4x4 (~2.3M params)
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 2. AlexNet for CIFAR-10
class AlexNetCIFAR(nn.Module):
    """
    AlexNet-style network for unresized CIFAR-10 inputs, shape (3, 32, 32).

    The feature extractor follows the AlexNet channel pattern with 3x3
    convolutions and stride-1 early layers, which is the common CIFAR-friendly
    adaptation of AlexNet's ImageNet architecture.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 -> 16

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16 -> 8

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_alexnet_cifar(num_classes=10):
    return AlexNetCIFAR(num_classes=num_classes)


def get_model(name, num_classes):
    name = name.lower()
    if name == 'mnist_mlp':
        return MNIST_MLP(num_classes=num_classes)
    if name == 'fcnet':
        return FCNet()
    if name == 'alexnet_cifar':
        return get_alexnet_cifar(num_classes=num_classes)
    if name.lower() == 'fcn':
        return FCN(num_classes=num_classes)
    raise ValueError("This MNIST demo supports model_name='mnist_mlp', 'fcnet', or 'alexnet_cifar'.")
