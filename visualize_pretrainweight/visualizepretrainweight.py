import torch
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np

# Load pre-trained model
model = models.resnet18(weights='DEFAULT')# or weights='IMAGENET1K_V1', or weights=None
model.eval()

# Get parameters of the first convolutional layer
conv1_weights = model.conv1.weight

# Visualize filter weights
num_filters = conv1_weights.size(0)
fig, axes = plt.subplots(num_filters // 8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(conv1_weights[i].detach().cpu().numpy().transpose(1, 2, 0))
    ax.axis('off')
plt.show()
#plt.savefig(f'pretrained_weights_resnet18_imagenet.jpg')  # Save the plot as an image file
#plt.close()

#visiualize without pretrain weight
model = models.resnet18(weights=None)
model.load_state_dict(torch.load('model_resnet18_withoutpretrain_state.pt'))
model.eval()

# Get parameters of the first convolutional layer
conv1_weights = model.conv1.weight

# Visualize filter weights
num_filters = conv1_weights.size(0)
fig, axes = plt.subplots(num_filters // 8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(conv1_weights[i].detach().cpu().numpy().transpose(1, 2, 0))
    ax.axis('off')

    # Enhance color saturation
    from PIL import ImageEnhance, Image

    # Chuyển đổi tensor PyTorch sang mảng NumPy và chuyển chiều cuối cùng thành định dạng HWC (Height x Width x Channels)
    conv1_weights_numpy = conv1_weights[i].detach().cpu().numpy().transpose(1, 2, 0)

    # Tăng cường các kênh màu cơ bản (red, green, blue)
    enhanced_image = conv1_weights_numpy.copy()

    # Tăng cường màu đỏ (chỉ tăng giá trị của kênh màu đỏ)
    enhanced_image[:, :, 0] = np.clip(3. * enhanced_image[:, :, 0], 0, 1)  # Tăng cường màu đỏ lên 1.5 lần

    # Tăng cường màu xanh lá cây (chỉ tăng giá trị của kênh màu xanh lá cây)
    enhanced_image[:, :, 1] = np.clip(3. * enhanced_image[:, :, 1], 0, 1)  # Tăng cường màu xanh lá cây lên 1.3 lần

    # Tăng cường màu xanh dương (chỉ tăng giá trị của kênh màu xanh dương)
    enhanced_image[:, :, 2] = np.clip(3. * enhanced_image[:, :, 2], 0, 1)  # Tăng cường màu xanh dương lên 1.2 lần

    # Giảm giá trị của các kênh màu để tăng cường màu đen
    black_factor = 0.7  # Giảm giá trị của các kênh màu xuống 80%
    enhanced_image[:, :, :3] = np.clip(enhanced_image[:, :, :3] * black_factor, 0, 1)

    # Chuyển đổi trở lại sang định dạng uint8 và tạo hình ảnh từ mảng NumPy
    enhanced_image = (enhanced_image * 255).astype('uint8')
    enhanced_image = Image.fromarray(enhanced_image)

    ax.imshow(enhanced_image)
# plt.show()
plt.savefig(f'withoutpretrained_weights_99.jpg')  # Save the plot as an image file
plt.close()

#visiualize with pretrain weight: run from train
model = models.resnet18(weights=None)
model.load_state_dict(torch.load('model_resnet18_usepretrain_state.pt'))
model.eval()

# Get parameters of the first convolutional layer
conv1_weights = model.conv1.weight

# Visualize filter weights
num_filters = conv1_weights.size(0)
fig, axes = plt.subplots(num_filters // 8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(conv1_weights[i].detach().cpu().numpy().transpose(1, 2, 0))
    ax.axis('off')
plt.show()

#visiualize with pretrain weight from Imagenet: download form google
model = models.resnet18(weights=None)
model.load_state_dict(torch.load('resnet18_dl.pth'))
model.eval()

# Get parameters of the first convolutional layer
conv1_weights = model.conv1.weight

# Visualize filter weights
num_filters = conv1_weights.size(0)
fig, axes = plt.subplots(num_filters // 8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(conv1_weights[i].detach().cpu().numpy().transpose(1, 2, 0))
    ax.axis('off')
plt.show()