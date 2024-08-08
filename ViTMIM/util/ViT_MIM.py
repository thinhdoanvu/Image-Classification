import torch
import torch.nn as nn


class MaskedImageModeling(nn.Module):
    def __init__(self, hidden_d, img_size, num_samples):
        super(MaskedImageModeling, self).__init__()
        self.hidden_d = hidden_d
        self.img_size = img_size
        self.num_samples = num_samples

        # Define the MIM network (simple MLP in this case)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, hidden_d * 2),  # Expand hidden dimensions
            nn.ReLU(),
            nn.Linear(hidden_d * 2, img_size * img_size * 3)  # Reconstruct the image
        )

    def forward(self, x):
        batch_size, num_patches, hidden_d = x.size()

        # Check if dimensions are correct
        if x.dim() != 3:
            raise ValueError(f"Expected input tensor to have 3 dimensions, but got {x.dim()} dimensions.")

        # Flatten the patches for the MLP
        x_flattened = x.view(-1, hidden_d)  # Shape: [batch_size * num_patches, hidden_d]

        # Process through the MIM network
        reconstructed_patches_flattened = self.mlp(x_flattened)  # Shape: [batch_size * num_patches, img_size * img_size * 3]

        # Reshape back to [batch_size, num_patches, 3, img_size, img_size]
        reconstructed_patches = reconstructed_patches_flattened.view(batch_size, num_patches, 3, self.img_size, self.img_size)

        return reconstructed_patches

    # Output: 3 x 28 x 28