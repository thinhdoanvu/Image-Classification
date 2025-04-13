import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose
from torchmetrics.functional import accuracy, precision, recall
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def setup_cuda(seed=50):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model():
    model.train()
    total_loss, total_acc, total_prec, total_rec = 0.0, 0.0, 0.0, 0.0

    for img, label in tqdm(train_loader, ncols=80, desc='Training'):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()

        logits = model(img)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        total_acc += accuracy(preds, label, task='multiclass', average='macro', num_classes=len(class_names)).item()
        total_prec += precision(preds, label, task='multiclass', average='macro', num_classes=len(class_names)).item()
        total_rec += recall(preds, label, task='multiclass', average='macro', num_classes=len(class_names)).item()

    n = len(train_loader)
    return total_loss/n, total_acc/n, total_prec/n, total_rec/n

def validate_model():
    model.eval()
    total_loss, total_acc, total_prec, total_rec = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for img, label in tqdm(val_loader, ncols=80, desc='Valid'):
            img, label = img.to(device), label.to(device)

            logits = model(img)
            loss = loss_fn(logits, label)

            preds = logits.argmax(dim=1)
            total_loss += loss.item()
            total_acc += accuracy(preds, label, task='multiclass', average='macro', num_classes=len(class_names)).item()
            total_prec += precision(preds, label, task='multiclass', average='macro', num_classes=len(class_names)).item()
            total_rec += recall(preds, label, task='multiclass', average='macro', num_classes=len(class_names)).item()

    n = len(val_loader)
    return total_loss/n, total_acc/n, total_prec/n, total_rec/n

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                 train_precisions, val_precisions, train_recalls, val_recalls,
                 save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.legend(); plt.title("Loss"); plt.xlabel("Epochs")

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.legend(); plt.title("Accuracy"); plt.xlabel("Epochs")

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_precisions, label='Train Precision')
    plt.plot(epochs, val_precisions, label='Val Precision')
    plt.legend(); plt.title("Precision"); plt.xlabel("Epochs")

    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_recalls, label='Train Recall')
    plt.plot(epochs, val_recalls, label='Val Recall')
    plt.legend(); plt.title("Recall"); plt.xlabel("Epochs")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ========== Main ==========
if __name__ == "__main__":
    device = setup_cuda()

    transform = Compose([Resize((224, 224)), ToTensor()])
    train_dataset = ImageFolder('../flowers102/train', transform=transform)
    val_dataset = ImageFolder('../flowers102/valid', transform=transform)
    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    import timm
    model = timm.create_model('resnet18', pretrained=True, num_classes=len(class_names)).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Metrics containers
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []

    best_val_acc = 0
    patience, counter = 30, 0

    os.makedirs("weights", exist_ok=True)
    best_model_path = "weights/resnet18_best_fit.pt"

    for epoch in range(200):
        print(f"\n📘 Epoch {epoch + 1}")
        tr_loss, tr_acc, tr_prec, tr_rec = train_model()
        val_loss, val_acc, val_prec, val_rec = validate_model()

        print(f"Train  | Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}, Prec: {tr_prec:.4f}, Rec: {tr_rec:.4f}")
        print(f"Val    | Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}")

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accuracies.append(tr_acc)
        val_accuracies.append(val_acc)
        train_precisions.append(tr_prec)
        val_precisions.append(val_prec)
        train_recalls.append(tr_rec)
        val_recalls.append(val_rec)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved: {best_model_path}")
        else:
            counter += 1
            print(f"⏸️ No improvement for {counter} epoch(s)")

        if counter >= patience:
            print("⛔ Early stopping triggered.")
            break

    # Save final training graph
    plot_metrics(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        train_precisions, val_precisions,
        train_recalls, val_recalls,
        save_path="outputs/resnet18_training_plot.png"
    )
