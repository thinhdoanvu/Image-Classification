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

############################Definine Functions##########################
train_data = './data/train'
valid_data = './data/valid'
n_epoch = 10

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

    print(f"Epoch:{epoch} train loss is {loss.item()}")
    return train_loss


def validate_model():
    """
    Validate the model over a single epoch
    :return: validation loss and segmentation performance
    """
    clf.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(valid_loader, ncols=80, desc='Testing'):  # valid_loader

            # Get a batch
            X, y = batch  # X: train, y = label
            X, y = X.to(device), y.to(device)

            # Perform a feed-forward pass
            logits = clf(X)  # yhat = predict

            # Compute the batch loss
            loss = loss_fn(logits, y)  # y:label

            # Accumulate the batch loss
            valid_loss += loss.item()

            # return valid_loss #valid loader
        print(f"Epoch:{epoch} valid loss is {loss.item()}")
        return valid_loss


if __name__ == "__main__":
    device = setup_cuda()

    # 1. Load the dataset
    from utils.getdataset import Getdataset
    transform = transforms.Compose([Resize((224, 224)), ToTensor()])
    train_dataset = Getdataset(train_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataset = Getdataset(valid_data, transform=transform)
    valid_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. Create a segmentation model
    import torchvision.models as models
    clf = models.resnet18(pretrained=True).to(device)

    # 3. Specify loss function and optimizer
    optimizer = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Training flow
    for epoch in range(n_epoch):  # train for n_epoch
        # 5.1. Train the model over a single epoch
        train_model()

        # 5.2. Validate the model
        validate_model()

        with open('model_state.pt', 'wb') as f:
            save(clf.state_dict(), f)

    # predict
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    img = Image.open('1.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))

    img = Image.open('2.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))

    img = Image.open('3.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))

    img = Image.open('4.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))

    img = Image.open('5.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))
