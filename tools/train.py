import argparse
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from dlcv.dataset import SubsetSTL10
from dlcv.models import CustomizableNetwork
from dlcv.utils import write_results_to_csv, save_model, get_transforms
from dlcv.training import train_and_evaluate_model

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # Define transformations for training and testing
    transform_train = get_transforms(train=True, horizontal_flip_prob=args.horizontal_flip_prob, rotation_degrees=args.rotation_degrees)
    transform_test = get_transforms(train=False)

    # Load datasets
    train_dataset = SubsetSTL10(root=args.data_root, split='train', transform=transform_train, download=True, subset_size=args.subset_size)
    test_dataset = SubsetSTL10(root=args.data_root, split='test', transform=transform_test, download=True)

    # Initialize training and test loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = CustomizableNetwork(
        conv_layers=args.conv_layers,
        filters_conv1=args.filters_conv1,
        filters_conv2=args.filters_conv2,
        filters_conv3=args.filters_conv3,
        dense_units=args.dense_units
    ).to(device)

    # Load pretrained weights if specified
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))

    # Freeze layers if set as argument
    if args.frozen_layers:
        for param in model.parameters():
            param.requires_grad = False

    # Setup Optimizer and strify learning rate if set as argument
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)

    # Define the criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Hand everything to the train and evaluate model function
    train_losses, test_losses, test_accuracies = train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, args.epochs, device, scheduler=scheduler)

    # Save results to CSV
    write_results_to_csv(args.results_csv + "/" + args.run_name, train_losses, test_losses, test_accuracies)

    # Save the model using the default folder
    if args.save_model_path:
        save_model(model, args.save_model_path + "/" + args.run_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the Customizable Network on STL10.')
    parser.add_argument('--run_name', type=str, default="run", help='Name of the current run')
    
    # Options
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for dataset')
    parser.add_argument('--subset_size', type=int, default=500, help='Number of samples to use from the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--conv_layers', type=int, default=3, help='Number of convolutional layers', choices=[1, 2, 3])
    parser.add_argument('--filters_conv1', type=int, default=16, help='Number of filters in the first conv layer')
    parser.add_argument('--filters_conv2', type=int, default=32, help='Number of filters in the second conv layer')
    parser.add_argument('--filters_conv3', type=int, default=64, help='Number of filters in the third conv layer')
    parser.add_argument('--dense_units', type=int, default=128, help='Number of units in the dense layer')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained weights file')
    parser.add_argument('--frozen_layers', action='store_true', help='Comma-separated list of layer names to freeze')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate for the optimizer')
    parser.add_argument('--lr_milestones', nargs='+', type=int, default=[50, 100], help='Milestones for MultiStepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for MultiStepLR scheduler')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for SGD optimizer')
    parser.add_argument('--horizontal_flip_prob', type=float, default=0.0, help='Probability of applying horizontal flip; 0 means no horizontal flip')
    parser.add_argument('--rotation_degrees', type=float, default=0.0, help='Max degrees to rotate; 0 means no rotation')
    parser.add_argument('--results_csv', type=str, default='./results', help='Directory to save the CSV file of training results')
    parser.add_argument('--save_model_path', type=str, default='./saved_models', help='Directory to save the trained model')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--do_early_stopping', action='store_true', help='Enable or disable early stopping')

    args = parser.parse_args()
    main(args)
