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
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Thiết lập biến cần thiết
train_dir = '../data/train'
test_dir = '../data/test'
valid_dir = '../data/valid'
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32
IMG_SIZE = 224
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
patch_size = 16


# Thiết lập thiết bị (GPU hoặc CPU)
def setup_cuda():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


device = setup_cuda()

# Thiết lập các transform để xử lý ảnh
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# Hàm dự đoán kết quả cho một hình ảnh
def predict_image(image_path, model, transform, class_names, device):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
    predicted_label = class_names[predicted_class.item()]
    return img, predicted_label


# Hàm chính để dự đoán các hình ảnh trong tập test
def test_model():
    # 1. Tải dữ liệu và lớp từ tập train
    transform = transforms.Compose([Resize((224, 224)), ToTensor()])
    train_dataset = ImageFolder(root='../dataset/train', transform=transform)
    test_dataset = ImageFolder(root='../dataset/test', transform=transform)
    # Get class names
    class_names = train_dataset.classes

    # 2. Tạo mô hình CCT và tải trạng thái từ checkpoint
    from model.cct import cct_14_7x2_224
    model = cct_14_7x2_224(
        num_classes=len(class_names)
    ).to(device)

    folder_checkpoint = 'cpts4playcards'  # Define the folder name
    file_checkpoint = os.path.join(folder_checkpoint, 'epoch_96_acc_0.8271.pt')  # lay epoch cuoi cung
    model.load_state_dict(torch.load(file_checkpoint, device))
    print('Model loaded from checkpoint.')
    # Ensure the output directory exists
    output_dir = "playcards"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Dự đoán kết quả cho mỗi hình ảnh trong tập test
    for image_path in tqdm(test_dataset.imgs, desc='Testing'):
        img, predicted_label = predict_image(image_path[0], model, manual_transforms, class_names, device)
        # plt.imshow(img)
        # plt.title(f'Predicted: {predicted_label}')
        # plt.show()
        
        # Convert the tensor image back to a PIL image if necessary
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        # Create a plot
        fig, ax = plt.subplots()

        # Set white background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Remove axis
        ax.axis('off')

        # Display the image
        ax.imshow(img)

        # Add the predicted label as the title
        ax.set_title(f'Predicted: {predicted_label}', fontsize=12, pad=10)

        # Save the figure
        image_basename = os.path.basename(image_path[0])
        image_name, image_ext = os.path.splitext(image_basename)
        output_image_path = os.path.join(output_dir, f"{image_name}_pred_{predicted_label}.png")

        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


if __name__ == '__main__':
    # Read the CSV file
    csv_file_path = '../dataset/dataset.csv'  # path to your CSV file
    test_model()
