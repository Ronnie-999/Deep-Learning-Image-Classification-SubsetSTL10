import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomizableNetwork(nn.Module):

    def __init__(self, conv_layers: int, filters_conv1: int, filters_conv2: int, filters_conv3: int,dense_units: int):
        super(CustomizableNetwork, self).__init__()

        # Ensure the number of convolutional layers is at least 2 and at most 3
        assert 2 <= conv_layers <= 3, "conv_layers must be 2 or 3"

        # Define convolutional layers
        self.layers = nn.ModuleDict()
        self.layers['conv1'] = nn.Conv2d(3, filters_conv1, kernel_size=3, padding=1)
        self.layers['conv2'] = nn.Conv2d(filters_conv1, filters_conv2, kernel_size=3, padding=1)
        if conv_layers == 3:
            self.layers['conv3'] = nn.Conv2d(filters_conv2, filters_conv3, kernel_size=3, padding=1)

        # Define batch normalization layers
        self.layers['bn1'] = nn.BatchNorm2d(filters_conv1)
        self.layers['bn2'] = nn.BatchNorm2d(filters_conv2)
        if conv_layers == 3:
            self.layers['bn3'] = nn.BatchNorm2d(filters_conv3)

        # Define maxpooling layers
        self.layers['pool1'] = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layers['pool2'] = nn.MaxPool2d(kernel_size=2, stride=2)
        if conv_layers == 3:
            self.layers['pool3'] = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        # Define fully connected layers
        self.layers['fc1'] = nn.Linear(filters_conv3 if conv_layers == 3 else filters_conv2, dense_units)
        self.layers['fc2'] = nn.Linear(dense_units, 10)  # Assuming 10 output classes

    def forward(self, x):
        # Apply the first convolutional layer
        x = self.layers['conv1'](x)
        x = self.layers['bn1'](x)
        x = F.relu(x)
        x = self.layers['pool1'](x)

        # Apply the second convolutional layer
        x = self.layers['conv2'](x)
        x = self.layers['bn2'](x)
        x = F.relu(x)
        x = self.layers['pool2'](x)

        # Apply the third convolutional layer if it exists
        if 'conv3' in self.layers:
            x = self.layers['conv3'](x)
            x = self.layers['bn3'](x)
            x = F.relu(x)
            x = self.layers['pool3'](x)

        # Adaptive pooling and flattening
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # Apply the fully connected layers
        x = self.layers['fc1'](x)
        x = F.relu(x)
        x = self.layers['fc2'](x)

        return x
