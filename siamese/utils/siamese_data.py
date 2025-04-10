import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SiameseDataset(Dataset):
    def __init__(self, image_pairs, labels, transform=None):
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return (img1, img2), label

class SiameseDataModule:
    def __init__(self, train_pairs, train_labels, val_pairs, val_labels, batch_size=32, img_size=(244, 244)):
        self.train_pairs = train_pairs
        self.train_labels = train_labels
        self.val_pairs = val_pairs
        self.val_labels = val_labels
        self.batch_size = batch_size
        self.img_size = img_size

        self.train_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

    def get_loaders(self):
        train_dataset = SiameseDataset(self.train_pairs, self.train_labels, transform=self.train_transform)
        val_dataset = SiameseDataset(self.val_pairs, self.val_labels, transform=self.val_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
