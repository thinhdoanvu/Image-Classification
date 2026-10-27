# train.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# -------------------- Configuration --------------------
train_data = "../dataset/train"
valid_data = "../dataset/valid"
n_epoch = 100
batch_size = 32
learning_rate = 1e-3
seed = 50

output_dir = "results"
checkpoint_dir = "checkpoint"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


def setup_cuda(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # benchmark can help fixed-size images; deterministic behavior may be reduced.
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Train", ncols=100, leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size_now = labels.size(0)
        total_loss += loss.item() * batch_size_now
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size_now

        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{100 * total_correct / total_samples:.2f}%"
        )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, desc="Valid"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    pbar = tqdm(loader, desc=desc, ncols=100, leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size_now = labels.size(0)
        total_loss += loss.item() * batch_size_now
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size_now

        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{100 * total_correct / total_samples:.2f}%"
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        np.array(all_labels),
        np.array(all_predictions),
    )


def save_training_plots(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["valid_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
    plt.show()
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(8, 5))
