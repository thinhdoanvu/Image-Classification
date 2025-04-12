import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.datasetloader import DatasetLoader
from utils.siamese_data import SiameseDataModule
from utils.models import LeNetBackbone, ResNet18Backbone
from utils.visualize import visualize_dataloader


def euclidean_distance(x1, x2):
    return torch.sqrt(torch.clamp(torch.sum((x1 - x2) ** 2, dim=1), min=1e-7))


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distances = euclidean_distance(output1, output2)
        loss = label * distances.pow(2) + (1 - label) * torch.clamp(self.margin - distances, min=0.0).pow(2)
        return torch.mean(loss)


def compute_metrics(labels, distances, threshold=0.5):
    preds = (distances < threshold).int()
    labels = labels.int()
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    return acc, prec, rec


def graph(train_losses, val_losses, accuracies, precisions, recalls, save_path="training_metrics.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    # Train vs Val Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='green')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Precision
    plt.subplot(2, 2, 3)
    plt.plot(epochs, precisions, label='Precision', color='purple')
    plt.title('Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    # Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, recalls, label='Recall', color='red')
    plt.title('Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training graph to {save_path}")



def main():
    # ======== 1. Cài đặt cấu hình ========
    INPUT_SIZE = (244, 244)
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 1e-3
    MARGIN = 1.0
    PATIENCE = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📌 Using device:", DEVICE)

    # ======== 2. Load dữ liệu ========
    train_dir = "../flowers102/train"
    valid_dir = "../flowers102/valid"
    loader = DatasetLoader(train_dir, valid_dir)
    train_pairs, train_labels = loader.create_pairs_on_training()
    val_pairs, val_labels = loader.create_pairs_on_validation()

    data_module = SiameseDataModule(
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=val_pairs,
        val_labels=val_labels,
        batch_size=BATCH_SIZE,
        img_size=INPUT_SIZE
    )

    train_loader, val_loader = data_module.get_loaders()

    # ======== 2'. Visualize ========
    visualize_dataloader(train_loader, num_pairs=5, title="Training Samples", save_path="outputs/train_pairs.png")
    visualize_dataloader(val_loader, num_pairs=4, title="Validation Pairs", save_path="outputs/val_pairs.png")

    # ======== 3. Khởi tạo mô hình ========
    model = ResNet18Backbone().to(DEVICE)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_save_path = "weights/resnet18_best_fit.pt"
    os.makedirs("weights", exist_ok=True)

    # ======== 4. Training loop ========
    train_losses = []
    val_losses = []
    accuracies = []
    precisions = []
    recalls = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for (img1, img2), labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training"):
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.float().to(DEVICE)

            optimizer.zero_grad()
            output1 = model(img1)
            output2 = model(img2)

            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ======== 5. Validation loop ========
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_distances = []

        with torch.no_grad():
            for (img1, img2), labels in tqdm(val_loader, desc="🔍 Validating"):
                img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.float().to(DEVICE)
                output1 = model(img1)
                output2 = model(img2)
                loss = criterion(output1, output2, labels)
                val_loss += loss.item()

                distances = euclidean_distance(output1, output2)
                all_labels.append(labels.cpu())
                all_distances.append(distances.cpu())

        val_loss /= len(val_loader)
        all_labels = torch.cat(all_labels)
        all_distances = torch.cat(all_distances)

        acc, prec, rec = compute_metrics(all_labels, all_distances)

        print(f"📊 Epoch {epoch + 1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Acc={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")

        # ======== 6. Model checkpoint ========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Model saved at {model_save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⏸️ No improvement ({epochs_no_improve}/{PATIENCE})")

        # ======== 7. Early stopping ========
        if epoch >= 10 and epochs_no_improve >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

        # ======== 8. Save to graph ========
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)

    # Vẽ biểu đồ sau khi training hoàn tất
    graph(train_losses, val_losses, accuracies, precisions, recalls, save_path="outputs/resnet18_training_plot.png")


if __name__ == "__main__":
    main()
