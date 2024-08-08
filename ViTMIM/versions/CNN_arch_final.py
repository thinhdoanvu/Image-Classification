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
from util.CNN_Encoder_Decoder import Autoencoder

# Display image
def imshow(img, title=''):
    img = img / 2 + 0.5  # Convert [0, 1]
    img = img.cpu()  # Move tensor to CPU
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)

def train_model():
    train_loss = 0.0
    model.train()

    for batch in tqdm(train_loader, ncols=80, desc='Training'):
        optimizer.zero_grad()
        img, _ = batch
        img = img.to(device)  # Move to GPU if available

        # Randomly mask 25% of the patches
        mask = torch.rand(img.size(0), 3, img.size(2), img.size(3), device=device) > 0.25
        masked_images = img * mask

        # Forward pass
        logits = model(masked_images)
        loss = loss_fn(logits, img)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)

def validate_model():
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, ncols=80, desc='Valid'):
            img, _ = batch
            img = img.to(device)

            # Randomly mask 25% of the patches
            mask = torch.rand(img.size(0), 3, img.size(2), img.size(3), device=device) > 0.25
            masked_images = img * mask

            # Forward pass
            logits = model(masked_images)
            loss = loss_fn(logits, img)

            valid_loss += loss.item()

    return valid_loss / len(val_loader)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ImageFolder(root='../data/train/', transform=transform)
    val_dataset = ImageFolder(root='../data/valid/', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Load model
    model = Autoencoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    for epoch in range(num_epochs):
        train_loss = train_model()
        val_loss = validate_model()

        print(f'Epoch {epoch}: Training loss = {train_loss}, Validation loss = {val_loss}')

        # Save model if validation loss decreases
        folder_path = 'checkpoints'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, f'epoch_{epoch}_loss_{val_loss:.4f}.pt')
        torch.save(model.state_dict(), file_path)

        # Display images after each epoch
        with torch.no_grad():
            data_iter = iter(train_loader)
            images, _ = next(data_iter)
            images = images.to(device)
            mask = torch.rand(images.size(0), 3, images.size(2), images.size(3), device=device) > 0.25
            masked_images = images * mask
            outputs = model(masked_images)

            plt.figure(figsize=(12, 6))

            plt.subplot(3, 4, 1)
            imshow(torchvision.utils.make_grid(images[:8]), title='Original Images')

            plt.subplot(3, 4, 2)
            imshow(torchvision.utils.make_grid(masked_images[:8]), title='Masked Images')

            plt.subplot(3, 4, 3)
            imshow(torchvision.utils.make_grid(outputs[:8]), title='Reconstructed Images')

            plt.show()

    print("Training complete.")
