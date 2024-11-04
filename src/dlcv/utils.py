import csv
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def load_pretrained_weights(network, weights_path, device):
    """
    Loads pretrained weights (state_dict) into the specified network.

    Args:
        network (nn.Module): The network into which the weights are to be loaded.
        weights_path (str or pathlib.Path): The path to the file containing the pretrained weights.
        device (torch.device): The device on which the network is running (e.g., 'cpu' or 'cuda').
    Returns:
        network (nn.Module): The network with the pretrained weights loaded and adjusted if necessary.
    """

    state_dict = torch.load(weights_path, map_location=device)
    
    # Create a new state_dict with only matching keys
    new_state_dict = {k: v for k, v in state_dict.items() if k in network.state_dict() and network.state_dict()[k].shape == v.shape}
    
    # Load the new state dict
    network.load_state_dict(new_state_dict, strict=False)
    
    return network


def freeze_layers(network, frozen_layers):
    """
    Freezes the specified layers of a network. Freezing a layer means its parameters will not be updated during training.

    Args:
        network (nn.Module): The neural network to modify.
        frozen_layers (list of str): A list of layer identifiers whose parameters should be frozen.
    """
    for name, param in network.named_parameters():
        if any(frozen_layer in name for frozen_layer in frozen_layers):
            param.requires_grad = False
    

def save_model(model, path):
    """
    Saves the model state_dict to a specified file.

    Args:
        model (nn.Module): The PyTorch model to save. Only the state_dict should be saved.
        path (str): The path where to save the model. Without the postifix .pth
    """
    torch.save(model.state_dict(), path + ".pth")

def get_stratified_param_groups(network, base_lr=0.001, stratification_rates=None):
    """
    Creates parameter groups with different learning rates for different layers of the network.

    Args:
        network (nn.Module): The neural network for which the parameter groups are created.
        base_lr (float): The base learning rate for layers not specified in stratification_rates.
        stratification_rates (dict): A dictionary mapping layer names to specific learning rates.

    Returns:
        param_groups (list of dict): A list of parameter group dictionaries suitable for an optimizer.
                                     Outside of the function this param_groups variable can be used like:
                                     optimizer = torch.optim.Adam(param_groups)
    """
    param_groups = []
    for name, param in network.named_parameters():
        # Extract the layer name without considering the parameter type (e.g., weight, bias)
        layer_name = name.rsplit('.', 1)[0]  
        lr = stratification_rates.get(layer_name, base_lr) if stratification_rates else base_lr
        param_group = {'params': param, 'lr': lr}
        param_groups.append(param_group)
    return param_groups

def get_transforms(train=True, horizontal_flip_prob=0.0, rotation_degrees=0.0):
    """
    Creates a torchvision transform pipeline for training and testing datasets. For training, augmentations
    such as horizontal flipping and random rotation can be included. For testing, only essential transformations
    like normalization and converting the image to a tensor are applied.

    Args:
        train (bool): Indicates whether the transform is for training or testing. If True, augmentations are applied.
        horizontal_flip_prob (float): Probability of applying a horizontal flip to the images. Effective only if train=True.
        rotation_degrees (float): The range of degrees for random rotation. Effective only if train=True.

    Returns:
        torchvision.transforms.Compose: Composed torchvision transforms for data preprocessing.
    """
    transform_list = []

    # Convert the input image to a PyTorch tensor
    transform_list.append(transforms.ToTensor())

    # Normalize the image
    transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    # Apply horizontal flip and random rotation only if in training mode
    if train:
        # Apply horizontal flip with a given probability
        if horizontal_flip_prob > 0.0:
            transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip_prob))

        # Apply random rotation within the specified degree range
        if rotation_degrees > 0.0:
            transform_list.append(transforms.RandomRotation(degrees=(-rotation_degrees, rotation_degrees)))

    # Compose all the transformations together
    return transforms.Compose(transform_list)

def write_results_to_csv(file_path, train_losses, test_losses, test_accuracies):
    """
    Writes the training and testing results to a CSV file.

    Args:
        file_path (str): Path to the CSV file where results will be saved. Without the postfix .csv
        train_losses (list): List of training losses.
        test_losses (list): List of testing losses.
        test_accuracies (list): List of testing accuracies.
    """
    file_path_with_extension = file_path + ".csv"
    file_exists = os.path.exists(file_path_with_extension)
    
    with open(file_path_with_extension, mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], test_losses[epoch], test_accuracies[epoch]])

def plot_multiple_losses_and_accuracies(model_data_list):
    """
    Plots training and testing losses and accuracies for multiple models.

    Args:
        model_data_list (list of dict): A list of dictionaries containing the following keys:
            - 'name' (str): The name of the model (for the legend)
            - 'train_losses' (list): Training losses per epoch
            - 'test_losses' (list): Testing losses per epoch
            - 'test_accuracies' (list): Testing accuracies per epoch
    """
    # Create subplots based on the number of model data
    num_models = len(model_data_list)
    fig, axs = plt.subplots(1, num_models, figsize=(8 * num_models, 6))

    # Plot each model data on its corresponding subplot
    for i, model_data in enumerate(model_data_list):
        axs[i].plot(model_data['train_losses'], label=f"{model_data['name']} Train Loss")
        axs[i].plot(model_data['test_losses'], label=f"{model_data['name']} Test Loss")
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')
        axs[i].set_title(f'Training and Testing Losses - {model_data["name"]}')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    for model_data in model_data_list:
        plt.plot(model_data['test_accuracies'], label=f"{model_data['name']} Test Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracies')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_samples_with_predictions(images, labels, predictions, class_names):
    """
    Plots a grid of images with labels and predictions, with dynamically adjusted text placement.

    Args:
        images (Tensor): Batch of images.
        labels (Tensor): True labels corresponding to the images.
        predictions (Tensor): Predicted labels for the images.
        class_names (list): List of class names indexed according to labels.
    """
    num_images = images.shape[0]
    num_cols = 5
    num_rows = int(np.ceil(num_images / num_cols))

    plt.figure(figsize=(15, 3 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.axis('off')
        plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[predictions[i]]}", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(labels, preds, class_names):
    """
    Plots a confusion matrix using ground truth labels and predictions.
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
