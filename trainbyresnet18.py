# Import dependencies
import torch
from torchvision import datasets
from tqdm import tqdm
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

############################Definine Functions##########################
train_data = './data/train'
valid_data = './data/valid'
n_epoch = 100


# Function to visualize the pre-trained weights of ResNet18 and save as an image file
def visualize_and_save_pretrained_weights(model, epoch):
    conv1_weights = model.conv1.weight.detach().cpu()  # Extract the weights of the first convolutional layer
    num_filters = conv1_weights.size(0)
    fig, axes = plt.subplots(num_filters // 8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(conv1_weights[i].numpy().transpose(1, 2, 0))
        ax.axis('off')
    folder_path = 'outputs'  # Define the folder name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it does not exist
    file_path = os.path.join(folder_path, f'pretrained_weights_epoch_{epoch}.jpg')  # Save the plot as an image file
    plt.savefig(file_path)
    plt.close()


# Setup CUDA
def setup_cuda():
    # Setting seeds for reproducibility
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_model():
    """
    Train the model over a single epoch
    :return: training loss and segmentation performance
    """
    train_loss = 0.0
    clf.train()

    for batch in tqdm(train_loader, ncols=80, desc='Training'):
        # Set the gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Get a batch
        X, y = batch  # X: train, y = label
        X, y = X.to(device), y.to(device)

        # Perform a feed-forward pass
        logits = clf(X)  # yhat = predict

        # Compute the batch loss
        loss = loss_fn(logits, y)  # y:label

        # Compute gradient of the loss fn w.r.t the trainable weights
        loss.backward()

        # Update the trainable weights
        optimizer.step()

        # Accumulate the batch loss
        train_loss += loss.item()

    # print("Epoch:{} train loss is {:.4f}".format(epoch,train_loss))
    return train_loss / len(train_loader)#hàm trả về loss trung bình trên 1 epoch (tích lũy từ các batch)


def validate_model():
    """
    Validate the model over a single epoch
    :return: validation loss and segmentation performance
    """
    clf.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(valid_loader, ncols=80, desc='Valid'):  # valid_loader

            # Get a batch
            X, y = batch  # X: train, y = label
            X, y = X.to(device), y.to(device)

            # Perform a feed-forward pass
            logits = clf(X)  # yhat = predict

            # Compute the batch loss
            loss = loss_fn(logits, y)  # y:label

            # Accumulate the batch loss
            valid_loss += loss.item()#hàm trả về loss trung bình trên 1 epoch (tích lũy từ các batch)

        # return valid_loss #valid loader
        # print("Epoch:{} valid loss is {:.4f}".format(epoch,valid_loss))
        return valid_loss / len(valid_loader)


if __name__ == "__main__":
    device = setup_cuda()

    # 1. Load the dataset
    from utils.getdataset import Getdataset

    transform = transforms.Compose([Resize((224, 224)), ToTensor()])
    train_dataset = Getdataset(train_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataset = Getdataset(valid_data, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    # 2. Create a segmentation model
    import torchvision.models as models

    clf = models.resnet18(weights='DEFAULT').to(device)  # or weights='IMAGENET1K_V1', weights=None

    # 3. Specify loss function and optimizer
    optimizer = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Training flow
    max_err = 1
    for epoch in range(n_epoch):  # train for n_epoch
        # 5.1. Train the model over a single epoch
        train_err = train_model()
        # train_model()

        # 5.2. Validate the model
        valid_err = validate_model()
        # validate_model()

        print("Epoch:{} training loss is {:.4f}".format(epoch, train_err))
        print("Epoch:{} valid loss is {:.4f}".format(epoch, valid_err))

        # Visualize pre-trained weights and save as an image file after each epoch
        visualize_and_save_pretrained_weights(clf, epoch)

        folder_path = 'checkpoint'  # Define the folder name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # Create the folder if it does not exist

        if valid_err < max_err:
            file_path = os.path.join(folder_path,'resnet18_epoch_' + str(epoch) + '_valoss' + '_{0:.4f}'.format(valid_err) + '.pt')
            with open(file_path, 'wb') as f:
                save(clf.state_dict(), f)
                max_err = valid_err