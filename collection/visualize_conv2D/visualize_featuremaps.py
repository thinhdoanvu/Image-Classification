import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# Define the CNN model
from utils.CNN import CNNModel
from utils.resnet18 import ResNet18
from utils.seresnet18 import SEResNet18
from utils.alexnetmodel import ImageClassifier
from utils.cbamresnet18 import CBAMResNet18
from utils.coordinateresnet18 import CAResNet18
from utils.eca_resnet18 import ECAResNet18


def plot_feature_maps(feature_maps, n_columns=8):
    for layer_index, feature_map in enumerate(feature_maps):
        # Number of feature maps (channels)
        n_features = feature_map.shape[1]

        # Size of each feature map
        size = feature_map.shape[2]

        # Create a grid to plot the feature maps
        n_rows = n_features // n_columns
        display_grid = torch.zeros((size * n_rows, size * n_columns))

        for i in range(n_rows):
            for j in range(n_columns):
                channel_image = feature_map[0, i * n_columns + j].cpu().numpy()

                # Normalize the channel image for better visualization
                channel_image -= channel_image.mean()
                if channel_image.std() > 0:
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[i * size : (i + 1) * size, j * size : (j + 1) * size] = channel_image

        # Display the grid
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_columns, scale * n_rows))
        plt.title(f'Layer {layer_index + 1}')
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()


if __name__ == "__main__":
    # Create the CNN model, RESNET18, SERESNET18
    # model = CNNModel()
    # model = ResNet18(num_classes=1)
    # model = SEResNet18(num_classes=1)
    # model = ImageClassifier(num_classes=1)
    # model = CBAMResNet18(num_classes=1)
    # model = CAResNet18(num_classes=1)
    model = ECAResNet18(num_classes=1)

    # Load and preprocess the image
    img_path = 'example.jpg'  # Replace with your image path
    img = Image.open(img_path)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    image = transform(img).unsqueeze(0)  # Add batch dimension

    # Images size

    print("input size: 3 x 224 x 224")
    outputs = model(image)
    # Print the shape of each feature map
    for count, feature_map in enumerate(outputs, start=1):
        print(f"Layer {count}: {feature_map.shape}")

    # CNN Visualization

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(100, 200))
    for i in range(len(processed)):
        plots = fig.add_subplot(len(processed), 4, i + 1)
        imgplot = plt.imshow(processed[i])
        plt.axis("off")
        plots.set_title(f"Layer{i}", fontsize=10)

    plt.savefig('ECA_CA_SE_Resnet18_fm.jpg', bbox_inches="tight", pad_inches=0)