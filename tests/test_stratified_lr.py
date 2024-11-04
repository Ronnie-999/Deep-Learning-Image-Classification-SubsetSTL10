import pytest
import torch
from torch import nn

from dlcv.utils import get_stratified_param_groups

def test_get_stratified_param_groups():
    # Define a network with identifiable layers
    class CustomNetwork(nn.Module):
        def __init__(self):
            super(CustomNetwork, self).__init__()
            self.layer1 = nn.Linear(2, 2)
            self.layer2 = nn.Linear(2, 2)
            self.layer3 = nn.Linear(2, 2)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x

    # Initialize the network
    model = CustomNetwork()

    # Define stratification rates for layers
    stratification_rates = {
        'layer1': 0.01,  # High learning rate for layer1
        'layer2': 0.005  # Lower learning rate for layer2
    }

    # Base learning rate for layers not specified
    base_lr = 0.001

    # Call the function
    param_groups = get_stratified_param_groups(model, base_lr, stratification_rates)

    # Check the parameter groups
    expected_lrs = {'layer1': 0.01, 'layer2': 0.005, 'layer3': 0.001}
    for group in param_groups:
        layer_name = next((name for name, param in model.named_parameters() if param is group['params']), None)
        expected_lr = next((lr for layer, lr in expected_lrs.items() if layer in layer_name), base_lr)
        assert group[
                   'lr'] == expected_lr, f"Mismatch in learning rates for {layer_name}: expected {expected_lr}, got {group['lr']}"
