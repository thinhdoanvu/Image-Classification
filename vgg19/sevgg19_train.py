import torch
from torch import nn, save, load
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy
from torchvision.transforms import ToTensor, Resize
import numpy as np
import os
import matplotlib.pyplot as plt

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
        train_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=len(class_names)).item()

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
            val_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=len(class_names)).item()

    return valid_loss / len(val_loader), val_acc / len(val_loader)


# Example plotting function

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    # Losses
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.yscale('log')  # Log scale can help for loss curves with large values

    # Accuracies
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='green')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig("trainplot_vgg19se.png")  # You can change the file name and format (e.g., .png, .jpg, .pdf)

    plt.show()


if __name__ == "__main__":
    device = setup_cuda()

    # 1. Load the dataset
    transform = transforms.Compose([Resize((224, 224)), ToTensor()])
    train_dataset = ImageFolder(root='../garbage/dataset_split/train', transform=transform)
    val_dataset = ImageFolder(root='../garbage/dataset_split/val', transform=transform)
    # Get class names
    class_names = train_dataset.classes

    # 2. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    # 3. Create a new deep model without pre-trained weights
    from utils.model import SEVGG19
    model = SEVGG19(
        num_classes=len(class_names),
    ).to(device)

    # 4. Specify loss function and optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 5. Train the model with 100 epochs
    # store the metrics for plotting
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    max_acc = 0
    for epoch in range(100):

        # 5.1. Train the model over a single epoch
        train_loss, train_acc = train_model()
        train_losses.append(train_loss)     # save train loss values
        train_accuracies.append(train_acc)  # save train acc values

        # 5.2. Validate the model after training
        val_loss, val_acc = validate_model()
        val_losses.append(val_loss)         # save val loss values
        val_accuracies.append(val_acc)      # save val acc values

        print(f'Epoch {epoch}: Train loss = {train_loss}, Train accuracy: {train_acc}')
        print(f'Epoch {epoch}: Validation loss = {val_loss}, Validation accuracy: {val_acc}')

        # 4.3. Save the model if the validation accuracy is increasing
        if val_acc > max_acc:
            print(f'Validation accuracy increased ({max_acc} --> {val_acc}). Model saved')
            folder_path = 'checkpoints_vgg19se'  # Define the folder name
            if os.path.exists(folder_path):
                pt_files = [f for f in os.listdir(folder_path) if f.endswith(".pt")]
                if pt_files:  # Nếu có file cũ, xóa nó
                    os.remove(os.path.join(folder_path, pt_files[0]))
            else:
                os.makedirs(folder_path)  # make new folder if not existing
            # path file to save
            file_path = os.path.join(folder_path, f'sevgg19_epoch_{epoch}_acc_{val_acc:.4f}.pt')

            # save model
            torch.save(model.state_dict(), file_path)

            max_acc = val_acc

# After training is complete, plot the metrics
plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)