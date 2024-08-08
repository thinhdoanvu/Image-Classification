import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Load backbone
# Import ConvStem and MyViT
from util.ConvStem import ConvStem
from util.myViT import *
from util.ViT_MIM import MaskedImageModeling


# Display image
def imshow(img, title=''):
    img = img / 2 + 0.5  # Convert [0, 1]
    img = img.cpu()  # Move tensor to CPU
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)


class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Adjust based on the output range of your images
        )

    def forward(self, x):
        return self.decoder(x)


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
    vit_model = MyViT(chw=(256, 28, 28),
                      n_patches=7,
                      hidden_d=8,
                      n_heads=2,
                      n_blocks=2,
                      num_samples=15).to(device)

    mim_model = MaskedImageModeling(hidden_d=8, img_size=28, num_samples=15).to(device)

    # Initialize the classifier
    input_dim = 14 * 14 * 96  # Based on the output size of Conv2D
    num_classes = 2  # Number of classes in your classification task
    classifier = SimpleClassifier(input_dim, num_classes).to(device)

    # Example of passing data through the network
    # Example of passing data through the network
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Step 1: Get feature maps from ConvStem
        feature_maps = conv_model(images)  # Output shape should be [batch_size, 256, 28, 28]
        print(f'Feature map size after ConvStem: {feature_maps.size()}')

        # Step 2: Pass feature maps through MyViT
        output_vit = vit_model(feature_maps)
        print(f'Output of ViT model: {output_vit.shape}')

        # Step 3: Masked Image Modeling from output ViT
        reconstructed_images = mim_model(output_vit)
        print(f'Reconstructed images size: {reconstructed_images.size()}')

        # Step 4: Downstream Conv2D
        batch_size, num_samples, _, img_size, _ = reconstructed_images.size()
        reconstructed_images = reconstructed_images.view(batch_size * num_samples, 3, img_size, img_size)
        conv_layer = nn.Conv2d(3, 96, 3, 2, 1).to(device)
        conv_output = conv_layer(reconstructed_images)
        print(f'Output size after Conv2D: {conv_output.size()}')

        # Step 5: Classification
        batch_size, channels, height, width = conv_output.size()
        flattened_conv_output = conv_output.view(batch_size, -1)  # Flatten for classifier

        # Labels need to be repeated to match the batch size of the classifier input
        # Each image has num_samples associated with it
        repeated_labels = labels.repeat_interleave(num_samples).to(device)

        outputs = classifier(flattened_conv_output)

        # Calculate loss and perform backpropagation
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        loss = criterion(outputs, repeated_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Classifier output size: {outputs.size()}')
        print(f'Loss: {loss.item()}')

        break  # Break after one batch for demonstration