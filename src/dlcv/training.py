import torch
from tqdm import tqdm

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """
    Trains a given model for one epoch using the provided data loader, criterion, and optimizer.

    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): The data loader providing the training data.
        criterion (nn.Module): The loss function to be used during training.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss per batch for the entire epoch.
    """
    model.train()
    total_loss = 0

    for data, labels in tqdm(data_loader, desc="Training"):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)

    average_loss = total_loss / len(data_loader.dataset)
    return average_loss

def evaluate_one_epoch(model, data_loader, criterion, device):
    """
    Tests a given model for one epoch using the provided data loader and criterion.

    Args:
        model (nn.Module): The model to be tested.
        data_loader (DataLoader): The data loader providing the testing data.
        criterion (nn.Module): The loss function to be used during testing.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss per batch for the entire epoch.
        float: The accuracy of the model on the test data.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Evaluating"):
            data, labels = data.to(device), labels.to(device)
            predictions = model(data)
            loss = criterion(predictions, labels)
            total_loss += loss.item() * data.size(0)
            _, predicted_labels = torch.max(predictions, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += data.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy

def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None, early_stopping=False):
    """
    Trains a given model for a specified number of epochs using the provided data loader, criterion,
    and optimizer, and tracks the loss for each epoch.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader providing the training data.
        test_loader (DataLoader): The data loader providing the testing data.
        criterion (nn.Module): The loss function to be used during training and testing.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        num_epochs (int): The number of epochs to train the model.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler (default is None).

    Returns:
        list: A list of the average loss per batch for each epoch.
        list: A list of the average loss per batch for each testing epoch.
        list: A list of the accuracy for each testing epoch.
    """
    train_losses = []
    test_losses = []
    accuracies = []
    consecutive_increases = 0  # Counter for consecutive increases in test loss

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, accuracy = evaluate_one_epoch(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        if scheduler:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.4f}")

        if early_stopping and epoch > 0 and test_loss >= test_losses[epoch - 1]:
            consecutive_increases += 1
            if consecutive_increases >= 2:
                print("Early stopping triggered. Stopping training.")
                break
        else:
            consecutive_increases = 0

    return train_losses, test_losses, accuracies