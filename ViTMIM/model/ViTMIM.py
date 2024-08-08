import torch
import torch.nn as nn
# Load backbone
from util.ConvStem import ConvStem
from util.myViT import *

# Define the SimpleClassifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

class Conv2dProcess(nn.Module):
    def __init__(self):
        super(Conv2dProcess, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 96, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(96, 128, 3, 2, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block(x)


class ViTMIM(nn.Module):
    def __init__(self, num_classes=2):  # Provide a default or specific number of classes
        super(ViTMIM, self).__init__()
        # Initialize models
        self.conv_model = ConvStem()
        self.encoder_model = ViTEncoder(
            chw=(256, 28, 28),
            n_patches=7,
            hidden_d=8,
            n_heads=2,
            n_blocks=2,
            num_samples=15)
        self.decoder_model = ViTDecoder(n_patches=7, hidden_d=8, n_heads=2, n_blocks=2)
        self.conv_downstream_model = Conv2dProcess()
        self.classifier = SimpleClassifier(input_dim=7 * 7 * 128, num_classes=num_classes)  # Initialize classifier

    def forward(self, images):
        # Step 1: Get feature maps from ConvStem
        feature_maps = self.conv_model(images)  # Output shape should be [batch_size, 256, 28, 28]

        # Step 2: Pass feature maps through Encoder
        output_encoder, indices = self.encoder_model(feature_maps)

        # Step 3: Reorder patches
        reordered_patches = reorder_patches(output_encoder, indices, self.encoder_model.n_patches ** 2)

        # Step 4: Decoder after reordering patches
        decoded_images = self.decoder_model(reordered_patches)  # Output: [batch_size, 49, 3, 4, 4]

        # Step 5: Reshape decoded images
        batch_size, num_patches, channels, patch_height, patch_width = decoded_images.size()
        patches_per_side = int(num_patches ** 0.5)  # Assuming a square grid of patches
        image_size = patches_per_side * patch_height  # Reconstructing image size
        reshaped_images = decoded_images.view(batch_size, channels, image_size, image_size)

        # Step 6: ConvDownstream
        downstream_output = self.conv_downstream_model(reshaped_images)

        # Step 7: Flatten for classification
        flattened_output = downstream_output.view(downstream_output.size(0), -1)  # Flatten for classifier

        # Step 8: Classifier
        final_output = self.classifier(flattened_output)

        return final_output
