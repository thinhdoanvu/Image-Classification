
"""
Created on 5 Aug 2024
@author: ThinhDV
Update MLP for Classification
"""

# Update Architecture: 3x224x224 - 3x112x112 - 3x28x28 - 49patches (choice 15/49) - Decoder (49 patches) = 3x28x28
# Feature map size after ConvStem: torch.Size([2, 256, 28, 28])
# Output of ViT model: torch.Size([2, 15, 8])
# Reordered patches shape: torch.Size([2, 49, 8])

# Decoded images size: torch.Size([2, 49, 3, 4, 4])
# Output: torch.Size([2, 49, 3, 4, 4]) of the decoded images means the following:
# 2: Batch size — there are 2 images in the batch.
# 49: Number of patches — each image is divided into 49 patches = 7 patches in row/col.
# 3: Number of channels — each patch has 3 channels (e.g., RGB).
# 4 x 4: Height and width of each patch — each patch has dimensions 4x4 pixels
# Image size: 3 x 28 x 28 = 3 x (4x7) x (4x7)

# Reshaped images size: torch.Size([2, 3, 28, 28])
# Output of ConvDownstream: torch.Size([2, 128, 7, 7])

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load backbone
from util.ConvStem import ConvStem
from util.myViT import *


# Define the SimpleClassifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


class Conv2dProcess(nn.Module):
    def __init__(self):
        super(Conv2dProcess, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 96, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(96, 128, 3, 2, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block(x)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        feature_maps = conv_model(images)
        output_encoder, indices = encoder_model(feature_maps)
        reordered_patches = reorder_patches(output_encoder, indices, encoder_model.n_patches ** 2)
        decoded_images = decoder_model(reordered_patches)
        reshaped_images = decoded_images.view(decoded_images.size(0), 3, 28, 28)  # Reshape for ConvDownstream
        conv_output = conv_downstream_model(reshaped_images)
        flattened_output = conv_output.view(conv_output.size(0), -1)  # Flatten for classifier

        outputs = classifier(flattened_output)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            feature_maps = conv_model(images)
            output_encoder, indices = encoder_model(feature_maps)
            reordered_patches = reorder_patches(output_encoder, indices, encoder_model.n_patches ** 2)
            decoded_images = decoder_model(reordered_patches)
            reshaped_images = decoded_images.view(decoded_images.size(0), 3, 28, 28)  # Reshape for ConvDownstream
            conv_output = conv_downstream_model(reshaped_images)
            flattened_output = conv_output.view(conv_output.size(0), -1)  # Flatten for classifier

            outputs = classifier(flattened_output)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = total_loss / len(val_loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy


if __name__ == '__main__':
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define datasets
    train_dataset = ImageFolder(root='../..//data/train/', transform=transform)
    val_dataset = ImageFolder(root='../../data/valid/', transform=transform)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize models
    conv_model = ConvStem().to(device)
    encoder_model = ViTEncoder(
        chw=(256, 28, 28),
        n_patches=7,
        hidden_d=8,
        n_heads=2,
        n_blocks=2,
        num_samples=15).to(device)

    decoder_model = ViTDecoder(n_patches=7, hidden_d=8, n_heads=2, n_blocks=2).to(device)
    conv_downstream_model = Conv2dProcess().to(device)  # Output of ConvDownstream: torch.Size([2, 128, 7, 7])
    input_dim = 7 * 7 * 128  # Based on the output size of Conv2d_Process
    num_classes = 2  # Number of classes in your classification task
    classifier = SimpleClassifier(input_dim, num_classes).to(device)

    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Training and validation loop
    num_epochs = 3
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Training
        train_loss, train_accuracy = train_model(classifier, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Validation
        val_loss, val_accuracy = validate_model(classifier, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    print('Training complete')
