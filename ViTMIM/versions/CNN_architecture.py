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


# Định nghĩa mạng autoencoder cơ bản
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Số kênh đầu vào là 3 cho hình ảnh màu
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid()  # Để đầu ra nằm trong khoảng [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Tạo dữ liệu và chuẩn bị data loader
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Kích thước ảnh có thể thay đổi theo yêu cầu
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Chỉnh sửa giá trị cho 3 kênh màu
])

train_dataset = ImageFolder(root='../data/train', transform=transform)
val_dataset = ImageFolder(root='../data/valid', transform=transform)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Khởi tạo mô hình, mất mát và optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Hàm để hiển thị ảnh
def imshow(img, title=''):
    img = img / 2 + 0.5  # Chuyển về khoảng [0, 1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# ---- MAIN ---- #
# Huấn luyện mô hình
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, ncols=80, desc=f'Training Epoch {epoch + 1}'):
        img, _ = batch
        # Randomly mask 25% of the patches
        mask = torch.rand(img.size(0), 3, img.size(2), img.size(3)) > 0.25
        masked_img = img * mask

        # Forward pass
        output = model(masked_img)
        loss = criterion(output, img)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    # Hiển thị một số hình ảnh sau mỗi epoch
    model.eval()
    with torch.no_grad():
        data_iter = iter(train_loader)
        images, _ = next(data_iter)
        mask = torch.rand(images.size(0), 3, images.size(2), images.size(3)) > 0.25
        masked_images = images * mask
        outputs = model(masked_images)

        # Hiển thị một số hình ảnh
        plt.figure(figsize=(12, 6))

        # Hiển thị ảnh gốc
        plt.subplot(3, 4, 1)
        imshow(torchvision.utils.make_grid(images[0:8]), title='Original Images')

        # Hiển thị ảnh bị mask
        plt.subplot(3, 4, 2)
        imshow(torchvision.utils.make_grid(masked_images[0:8]), title='Masked Images')

        # Hiển thị ảnh sau khi huấn luyện
        plt.subplot(3, 4, 3)
        imshow(torchvision.utils.make_grid(outputs[0:8]), title='Reconstructed Images')

        plt.show()

print("Training complete.")
