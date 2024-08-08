
"""
Created on 4 Aug 2024
@author: ThinhDV
Use MaskedImageModeling for recover and Conv2dProcess for Classification
Resuts as ver3,4 and ViTMIM model
"""

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
from util.ViT_MIM import MaskedImageModeling


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


def train_model(train_loader, conv_model, vit_model, mim_model, classifier, criterion, optimizer, device):
    conv_model.train()
    vit_model.train()
    mim_model.train()
    classifier.train()

    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Step 1: Get feature maps from ConvStem
        feature_maps = conv_model(images)

        # Step 2: Pass feature maps through MyViT
        output_encoder = encoder_model(feature_maps)

        # Step 3: Masked Image Modeling from output ViT
        reconstructed_images = mim_model(output_encoder)

        # Step 4: Downstream Conv2D
        batch_size, num_samples, _, img_size, _ = reconstructed_images.size()
        reconstructed_images = reconstructed_images.view(batch_size * num_samples, 3, img_size, img_size)

        conv_output = conv2d_processor(reconstructed_images)
        #print(f'Output size after Conv2D processing: {conv_output.size()}')

        # Step 5: Classification
        batch_size, channels, height, width = conv_output.size()
        flattened_conv_output = conv_output.view(batch_size, -1)  # Flatten for classifier

        # Labels need to be repeated to match the batch size of the classifier input
        repeated_labels = labels.repeat_interleave(num_samples).to(device)

        outputs = classifier(flattened_conv_output)

        # Calculate loss and perform backpropagation
        loss = criterion(outputs, repeated_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Training loss: {epoch_loss:.4f}')
    return epoch_loss


def validate_model(val_loader, conv_model, vit_model, mim_model, classifier, criterion, device):
    conv_model.eval()
    vit_model.eval()
    mim_model.eval()
    classifier.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Step 1: Get feature maps from ConvStem
            feature_maps = conv_model(images)

            # Step 2: Pass feature maps through MyViT
            output_encoder = encoder_model(feature_maps)

            # Step 3: Masked Image Modeling from output ViT
            reconstructed_images = mim_model(output_encoder)

            # Step 4: Downstream Conv2D
            batch_size, num_samples, _, img_size, _ = reconstructed_images.size()
            reconstructed_images = reconstructed_images.view(batch_size * num_samples, 3, img_size, img_size)

            conv_output = conv2d_processor(reconstructed_images)
            print(f'Output size after Conv2D processing: {conv_output.size()}')

            # Step 5: Classification
            batch_size, channels, height, width = conv_output.size()
            flattened_conv_output = conv_output.view(batch_size, -1)  # Flatten for classifier

            # Labels need to be repeated to match the batch size of the classifier input
            repeated_labels = labels.repeat_interleave(num_samples).to(device)

            outputs = classifier(flattened_conv_output)

            # Calculate loss
            loss = criterion(outputs, repeated_labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += repeated_labels.size(0)
            correct += (predicted == repeated_labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return epoch_loss, accuracy


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
    train_dataset = ImageFolder(root='../data/train/', transform=transform)
    val_dataset = ImageFolder(root='../data/valid/', transform=transform)

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

    mim_model = MaskedImageModeling(hidden_d=8, img_size=28, num_samples=15).to(device)

    input_dim = 7 * 7 * 128  # Based on the output size of Conv2d_Process
    num_classes = 2  # Number of classes in your classification task
    classifier = SimpleClassifier(input_dim, num_classes).to(device)

    # Initialize the classifier
    input_dim = 7 * 7 * 96  # Based on the output size of Conv2d_Process
    num_classes = 2  # Number of classes in your classification task
    # classifier = SimpleClassifier(input_dim, num_classes).to(device)
    conv2d_processor = Conv2dProcess().to(device)

    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Training and validation loop
    num_epochs = 1
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Training
        train_loss = train_model(train_loader, conv_model, encoder_model, mim_model, classifier, criterion, optimizer, device)

        # Validation
        val_loss, val_accuracy = validate_model(val_loader, conv_model, encoder_model, mim_model, classifier, criterion, device)

        # Save the best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(classifier.state_dict(), 'best_classifier.pth')
        #     print('Saved the best model')

    print('Training complete')
