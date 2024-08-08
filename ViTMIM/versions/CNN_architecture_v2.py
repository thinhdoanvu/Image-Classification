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
# define autoencoder by CNN
from util.CNN_Encoder_Decoder import Autoencoder


# display image
def imshow(img, title=''):
    img = img / 2 + 0.5  # convert [0, 1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def train_model():
    """
    Train the model over a single epoch
    :return: training loss and segmentation performance
    """
    train_loss = 0.0
    train_acc = 0.0
    model.train()

    for batch in tqdm(train_loader, ncols=80, desc='Training'):
        # Set the gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Get a batch
        img, _ = batch
        # Randomly mask 25% of the patches
        mask = torch.rand(img.size(0), 3, img.size(2), img.size(3)) > 0.25
        masked_img = img * mask

        # Perform a feed-forward pass
        logits = model(masked_img)

        # Compute the batch loss
        loss = loss_fn(logits, img)  # img:origin

        # Compute gradient of the loss fn w.r.t the trainable weights
        loss.backward()

        # Update the trainable weights
        optimizer.step()

        # Accumulate the batch loss
        train_loss += loss.item()  # count loss for all batch

        return train_loss / len(train_loader), train_acc / len(train_loader)


def validate_model():
    """
    Validate the model over a single epoch
    :return: validation loss and segmentation performance
    """
    model.eval()
    valid_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, ncols=80, desc='Valid'):  # valid_loader

            # Get a batch
            img, _ = batch
            # Randomly mask 25% of the patches
            mask = torch.rand(img.size(0), 3, img.size(2), img.size(3)) > 0.25
            masked_img = img * mask

            # Perform a feed-forward pass
            logits = model(masked_img)

            # Compute the batch loss
            loss = loss_fn(logits, img)  # img:origin

            # Accumulate the batch loss
            valid_loss += loss.item()  # count loss for all batch

        return valid_loss / len(val_loader), val_acc / len(val_loader)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # 1. read data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # image size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Chỉnh sửa giá trị cho 3 kênh màu
    ])

    train_dataset = ImageFolder(root='../data/train', transform=transform)
    val_dataset = ImageFolder(root='../data/valid', transform=transform)

    # data loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 2. load model, loss and optimizer
    model = Autoencoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. training
    num_epochs = 1
    max_acc = 0
    for epoch in range(num_epochs):
        # 3.1. Train the model over a single epoch
        train_loss, train_acc = train_model()

        # 3.2. Validate the model after training
        val_loss, val_acc = validate_model()

        print(f'Epoch {epoch}: Validation loss = {val_loss}, Validation accuracy: {val_acc}')

        # 3.3. Save the model if the validation accuracy is increasing
        if val_acc > max_acc:
            print(f'Validation accuracy increased ({max_acc} --> {val_acc}). Model saved')
            folder_path = 'checkpoints'  # Define the folder name
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)  # Create the folder if it does not exist
            file_path = os.path.join(folder_path,
                                     'epoch_' + str(epoch) + '_acc_{0:.4f}'.format(val_acc) + '.pt')
            with open(file_path, 'wb') as f:
                save(model.state_dict(), f)
            max_acc = val_acc

        # 4. display images after each epoch
        with torch.no_grad():
            data_iter = iter(train_loader)
            images, _ = next(data_iter)
            mask = torch.rand(images.size(0), 3, images.size(2), images.size(3)) > 0.25
            masked_images = images * mask
            outputs = model(masked_images)

            # frame
            plt.figure(figsize=(12, 6))

            # origin
            plt.subplot(3, 4, 1)
            imshow(torchvision.utils.make_grid(images[0:8]), title='Original Images')

            # mask
            plt.subplot(3, 4, 2)
            imshow(torchvision.utils.make_grid(masked_images[0:8]), title='Masked Images')

            # output
            plt.subplot(3, 4, 3)
            imshow(torchvision.utils.make_grid(outputs[0:8]), title='Reconstructed Images')

            plt.show()

    print("Training complete.")
