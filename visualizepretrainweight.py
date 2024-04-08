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
# plt.show()
plt.savefig(f'pretrained_weights_resnet18_imagenet.jpg')  # Save the plot as an image file
plt.close()

#visiualize pretrain weight from Imagenet: download it
# model = models.resnet18(weights=None)
# model.load_state_dict(torch.load('resnet18_dl.pth'))
# model.eval()
#
# # Get parameters of the first convolutional layer
# conv1_weights = model.conv1.weight
#
# # Visualize filter weights
# num_filters = conv1_weights.size(0)
# fig, axes = plt.subplots(num_filters // 8, 8, figsize=(12, 12))
# for i, ax in enumerate(axes.flatten()):
#     ax.imshow(conv1_weights[i].detach().cpu().numpy().transpose(1, 2, 0))
#     ax.axis('off')
# plt.show()