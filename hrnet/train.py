"""
Thanh Le  16 April 2024
How to train/fine-tune a pre-trained model on a custom dataset (i.e., transfer learning)
"""
import torch
from torch import nn, save, load
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy
from torchvision.transforms import ToTensor, Resize
import pandas as pd
import numpy as np
import os

def get_num_classes_from_csv(csv_file_path):
    # Đọc file CSV vào DataFrame
    df = pd.read_csv(csv_file_path)

    # Đếm số lớp duy nhất trong cột nhãn
    num_classes = df['labels'].nunique()

    return num_classes


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
    :return: training loss and training accuracy
    """
    train_loss = 0.0
    train_acc = 0.0
    model.train()

    for (img, label) in tqdm(train_loader, ncols=80, desc='Training'):
        # Get a batch
        img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)

        # Set the gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Perform a feed-forward pass
        logits = model(img)

        # Compute the batch loss
        loss = loss_fn(logits, label)

        # Compute gradient of the loss fn w.r.t the trainable weights
        loss.backward()

        # Update the trainable weights
        optimizer.step()

        # Accumulate the batch loss
        train_loss += loss.item()

        # Get the predictions to calculate the accuracy for every iteration. Remember to accumulate the accuracy
        prediction = logits.argmax(axis=1)
        train_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=num_classes).item()

    return train_loss / len(train_loader), train_acc / len(train_loader)


def validate_model():
    """
    Validate the model over a single epoch
    :return: validation loss and validation accuracy
    """
    model.eval()
    valid_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for (img, label) in tqdm(val_loader, ncols=80, desc='Valid'):
            # Get a batch
            img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)

            # Perform a feed-forward pass
            logits = model(img)

            # Compute the batch loss
            loss = loss_fn(logits, label)

            # Accumulate the batch loss
            valid_loss += loss.item()

            # Get the predictions to calculate the accuracy for every iteration. Remember to accumulate the accuracy
            prediction = logits.argmax(axis=1)
            val_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=num_classes).item()

    return valid_loss / len(val_loader), val_acc / len(val_loader)


if __name__ == "__main__":

    device = setup_cuda()

    # Đường dẫn đến file CSV
    csv_file_path = '../dataset/dataset.csv'

    # Xác định số lớp từ file CSV
    num_classes = get_num_classes_from_csv(csv_file_path)
    print("number of classes:", num_classes)

    # 1. Load the dataset
    transform = transforms.Compose([Resize((256, 256)), ToTensor()])
    train_dataset = ImageFolder(root='../dataset/train', transform=transform)
    val_dataset = ImageFolder(root='../dataset/test', transform=transform)

    # 2. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # 3. Create a new deep model with pre-trained weights
    import timm

    # 4. Note that the model pre-trained model has 1,000 output neurons (because ImageNet has 1,000 classes), so we must
    # customize the last linear layer to adapt to our 53-class problem (i.e., Cat vs Dog)
    model = timm.create_model('hrnet_w18', pretrain=True, num_classes=num_classes).to(device)

    # 4. Specify loss function and optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 5. Train the model with 100 epochs
    max_acc = 0
    folder_path = 'checkpoints'  # Define the folder name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it does not exist

    for epoch in range(100):

        # 5.1. Train the model over a single epoch
        train_loss, train_acc = train_model()

        # 5.2. Validate the model after training
        val_loss, val_acc = validate_model()

        print(f'Epoch {epoch}: Validation loss = {val_loss}, Validation accuracy: {val_acc}')

        # 4.3. Save the model if the validation accuracy is increasing
        if val_acc > max_acc:
            print(f'Validation accuracy increased ({max_acc} --> {val_acc}). Model saved')
            file_path = os.path.join(folder_path,'hrnet_epoch_' + str(epoch) + '_acc_{0:.4f}'.format(max_acc) + '.pt')
            with open(file_path, 'wb') as f:
                save(model.state_dict(), f)
                max_acc = val_acc
