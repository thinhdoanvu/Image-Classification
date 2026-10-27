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
train_data = "/kaggle/input/datasets/thnhonv/catdog/dataset/training_set"
valid_data = "/kaggle/input/datasets/thnhonv/catdog/dataset/test_set"
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
    plt.plot(epochs, [x * 100 for x in history["train_acc"]], label="Train accuracy")
    plt.plot(epochs, [x * 100 for x in history["valid_acc"]], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=200)
    plt.show()
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # For many classes, a large figure is easier to read.
    fig_size = max(8, min(24, len(class_names) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",
        colorbar=False
    )
    ax.set_title("Confusion Matrix (Validation Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.show()
    plt.close(fig)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        digits=4
    )
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)
    print("\nClassification report:\n")
    print(report)


if __name__ == "__main__":
    device = setup_cuda(seed)
    print("Device:", device)

    from utils.getdataset import Getdataset
    from utils.alexnetmodel import Alexnet

    transform = transforms.Compose([
        Resize((224, 224)),
        ToTensor(),
    ])

    train_dataset = Getdataset(train_data, transform=transform)
    valid_dataset = Getdataset(valid_data, transform=transform)

    # ImageFolder class order must match between train and validation sets.
    train_classes = train_dataset.data.classes
    valid_classes = valid_dataset.data.classes
    if train_classes != valid_classes:
        raise ValueError(
            "Training and validation classes/order do not match.\n"
            f"Train: {train_classes}\nValid: {valid_classes}"
        )

    class_names = train_classes
    num_class = len(class_names)
    print("Number of training images:", len(train_dataset))
    print("Number of validation images:", len(valid_dataset))
    print("Classes:", class_names)
    print("Class-to-index mapping:", train_dataset.data.class_to_idx)
    print("Number of classes:", num_class)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )


    model = Alexnet(num_class).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
    }
    best_valid_loss = float("inf")
    best_path = os.path.join(checkpoint_dir, "alexmodel_best.pt")

    for epoch in range(n_epoch):
        print(f"\nEpoch {epoch + 1}/{n_epoch}")

        train_loss, train_acc = run_train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        valid_loss, valid_acc, _, _ = evaluate(
            model, valid_loader, loss_fn, device, desc="Valid"
        )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

        print(
            f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc * 100:.2f}%"
        )
        print(
            f"Valid loss: {valid_loss:.4f} | Valid accuracy: {valid_acc * 100:.2f}%"
        )

        # Save the best model based on validation loss.
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_acc,
                    "class_names": class_names,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}")

        # Update learning curves as training proceeds.
        save_training_plots(history, output_dir)

    # Load best checkpoint and produce final validation metrics.
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"\nLoaded best checkpoint from epoch {checkpoint['epoch']} "
        f"(validation loss={checkpoint['valid_loss']:.4f})"
    )

    _, final_acc, y_true, y_pred = evaluate(
        model, valid_loader, loss_fn, device, desc="Final validation"
    )
    print(f"Best-checkpoint validation accuracy: {final_acc * 100:.2f}%")

    save_confusion_matrix(y_true, y_pred, class_names, output_dir)
    print(f"\nPlots and report saved in: {output_dir}/")
