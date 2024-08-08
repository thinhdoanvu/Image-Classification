
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

    decoder_model = ViTDecoder(n_patches=7, hidden_d=8, n_heads=2, n_blocks=2).to(device)
    conv_downstream_model = Conv2dProcess().to(device)

    # Example of passing data through the network
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Step 1: Get feature maps from ConvStem
        feature_maps = conv_model(images)  # Output shape should be [batch_size, 256, 28, 28]
        print(f'Feature map size after ConvStem: {feature_maps.size()}')

        # Step 2: Pass feature maps through Encoder
        output_encoder, indices = encoder_model(feature_maps)
        print(f'Output of ViT model: {output_encoder.shape}')

        # Step 3: Reorder patches
        reordered_patches = reorder_patches(output_encoder, indices, encoder_model.n_patches ** 2)
        print(f'Reordered patches shape: {reordered_patches.shape}')

        # Step 4: Decoder after reordering patches
        decoded_images = decoder_model(reordered_patches)
        print(f'Decoded images size: {decoded_images.size()}')

        # Output: torch.Size([2, 49, 3, 4, 4]) of the decoded images means the following:
        # 2: Batch size — there are 2 images in the batch.
        # 49: Number of patches — each image is divided into 49 patches = 7 patches in row/col.
        # 3: Number of channels — each patch has 3 channels (e.g., RGB).
        # 4 x 4: Height and width of each patch — each patch has dimensions 4x4 pixels
        # Image size: 3 x 28 x 28 = 3 x (4x7) x (4x7)

        # Step 5: Reshape decoded images
        batch_size, num_patches, channels, patch_height, patch_width = decoded_images.size()
        patches_per_side = int(num_patches ** 0.5)  # Assuming a square grid of patches
        image_size = patches_per_side * patch_height  # Reconstructing image size
        reshaped_images = decoded_images.view(batch_size, channels, image_size, image_size)
        print(f'Reshaped images size: {reshaped_images.size()}')

        # Step 6: ConvDownstream
        downstream_output = conv_downstream_model(reshaped_images)
        print(f'Output of ConvDownstream: {downstream_output.size()}')


        break  # Break after one batch for demonstration
