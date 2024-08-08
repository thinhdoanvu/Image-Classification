import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch


def patchify(feature_maps, n_patches):
    """
    Convert feature maps into patches.
    :param feature_maps: Tensor of shape [batch_size, channels, height, width]
    :param n_patches: Number of patches per row or column in each feature map
    :return: Tensor of shape [batch_size, num_patches, patch_dim]
    """
    n, c, h, w = feature_maps.shape

    # Check if dimensions are divisible by the number of patches
    assert h % n_patches == 0 and w % n_patches == 0, "Feature map dimensions should be divisible by n_patches"

    patch_size = h // n_patches  # Size of each patch

    # Compute number of patches
    num_patches = n_patches ** 2

    # Create a tensor to hold the patches
    patches = torch.zeros(n, num_patches, c * patch_size * patch_size)

    # Extract patches
    for idx in range(n):
        patches_list = []
        for i in range(n_patches):
            for j in range(n_patches):
                patch = feature_maps[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                patches_list.append(patch.flatten())
        patches[idx] = torch.stack(patches_list)

    return patches


def patchify_randomly(feature_maps, n_patches, num_samples):
    n, c, h, w = feature_maps.shape
    # Check if dimensions are divisible by the number of patches
    assert h % n_patches == 0 and w % n_patches == 0, "Feature map dimensions should be divisible by n_patches"
    patch_size = h // n_patches  # Size of each patch
    patch_dim = c * patch_size * patch_size  # Dimension of each patch
    # Compute number of patches
    total_patches = n_patches ** 2
    # Create a tensor to hold the patches
    patches = torch.zeros(n, total_patches, patch_dim)
    # Extract patches
    for idx in range(n):
        patches_list = []
        for i in range(n_patches):
            for j in range(n_patches):
                patch = feature_maps[idx, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                patches_list.append(patch.flatten())
        patches[idx] = torch.stack(patches_list)
    # Randomly sample patches
    if num_samples < total_patches:
        indices = torch.randperm(total_patches)[:num_samples]
        patches = patches[:, indices]
    else:
        indices = torch.arange(total_patches)
    return patches, indices


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d, dtype=torch.float32)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d))) if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


class EncoderBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(EncoderBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.cat(seq_result, dim=-1))  # Use torch.cat instead of torch.hstack
        return torch.stack(result)


class ViTEncoder(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7, hidden_d=8, n_heads=2, n_blocks=2, num_samples=15):
        super(ViTEncoder, self).__init__()
        self.chw = chw
        self.n_patches = n_patches
        self.hidden_d = hidden_d
        self.num_samples = num_samples
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"

        self.patch_size = chw[1] // n_patches
        self.input_d = chw[0] * self.patch_size * self.patch_size
        self.linear_mapper = nn.Linear(self.input_d, hidden_d)
        self.class_token = nn.Parameter(torch.rand(1, hidden_d))
        self.pos_embed = nn.Parameter(
            torch.tensor(get_positional_embeddings(n_patches ** 2 + 1, hidden_d), dtype=torch.float32))
        self.pos_embed.requires_grad = False
        self.blocks = nn.ModuleList([EncoderBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, hidden_d),
            nn.GELU(),
            nn.Linear(hidden_d, num_samples * hidden_d)  # Adjust to output [batch_size, num_samples, hidden_d]
        )

    def forward(self, feature_maps):
        # Extract and randomly sample patches
        patches, indices = patchify_randomly(feature_maps, n_patches=self.n_patches, num_samples=self.num_samples)

        # Move patches to the same device as the model, use for GPU
        patches = patches.to(self.linear_mapper.weight.device)

        tokens = self.linear_mapper(patches)
        tokens = torch.cat([self.class_token.expand(tokens.size(0), 1, -1), tokens], dim=1)

        # Ensure positional embedding has the correct size
        pos_embed = self.pos_embed[:tokens.size(1), :].unsqueeze(0).expand(tokens.size(0), -1, -1)

        out = tokens + pos_embed
        for block in self.blocks:
            out = block(out)
        out = out[:, 0]  # Extract the class token
        out = self.mlp(out)  # Get the final output

        # Reshape to [batch_size, num_samples, hidden_d]
        out = out.view(-1, self.num_samples, self.hidden_d)
        return out, indices


class FFNN(nn.Module):

    def __init__(
            self,
            d,
            bias=False,
            dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d, d, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)  # [B, T, 4*d]
        x = self.gelu(x)
        x = self.c_proj(x)  # [B, T, d]
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(DecoderBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.ffnn = FFNN(hidden_d)

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.ffnn(self.norm2(out))
        return out


def reorder_patches(encoded_patches, sampled_indices, total_patches):
    """
    Reorder patches based on the original indices.
    :param encoded_patches: Tensor of shape [batch_size, num_samples, hidden_d]
    :param sampled_indices: Indices of the sampled patches
    :param total_patches: Total number of patches before sampling
    :return: Reordered patches tensor
    """
    # Initialize tensor for all patches
    reordered_patches = torch.zeros(encoded_patches.size(0), total_patches, encoded_patches.size(2))

    # Place encoded patches in their original positions
    reordered_patches[:, sampled_indices] = encoded_patches

    return reordered_patches


class ViTDecoder(nn.Module):    # Decoded images size: torch.Size([2, 49, 3, 4, 4])
    def __init__(self, chw=(3, 28, 28), n_patches=7, hidden_d=8, n_heads=2, n_blocks=2):
        super(ViTDecoder, self).__init__()
        self.chw = chw
        self.n_patches = n_patches
        self.hidden_d = hidden_d
        self.patch_size = chw[1] // n_patches  # Compute patch size
        self.input_d = chw[0] * self.patch_size * self.patch_size
        self.linear_mapper = nn.Linear(hidden_d, self.input_d)
        self.class_token = nn.Parameter(torch.rand(1, hidden_d))
        self.pos_embed = nn.Parameter(
            torch.tensor(get_positional_embeddings(n_patches ** 2 + 1, hidden_d), dtype=torch.float32))
        self.pos_embed.requires_grad = False
        self.blocks = nn.ModuleList([DecoderBlock(hidden_d, n_heads) for _ in range(n_blocks)])

    def forward(self, feature_maps):
        batch_size, num_patches, _ = feature_maps.shape

        # Assuming feature_maps have shape [batch_size, num_patches, hidden_d]
        tokens = feature_maps

        tokens = torch.cat([self.class_token.expand(batch_size, 1, -1), tokens], dim=1)

        # Ensure positional embedding has the correct size
        pos_embed = self.pos_embed[:tokens.size(1), :].unsqueeze(0).expand(batch_size, -1, -1)

        out = tokens + pos_embed
        for block in self.blocks:
            out = block(out)

        # The output should be reshaped to the image dimensions
        # Use linear mapper to get the original image size
        out = self.linear_mapper(out)
        out = out[:, 1:]  # Remove the class token

        # Reshape to [batch_size, num_patches, channels, patch_height, patch_width]
        patch_size = self.chw[1] // self.n_patches
        out = out.view(batch_size, num_patches, self.chw[0], patch_size, patch_size)

        return out


# Using for test VIT model, don't worry and concern this
# Example usage
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Create a batch of dummy feature maps
    feature_maps = torch.randn(2, 256, 28, 28).to(device)  # Batch of 2 images

    # Initialize MyViT model
    # Initialize ViTEncoder model
    encoder_model = ViTEncoder(
        chw=(256, 28, 28),
        n_patches=7,
        hidden_d=8,
        n_heads=2,
        n_blocks=2,
        num_samples=15
    ).to(device)

    # Step 1: Get feature maps from ConvStem
    feature_maps = feature_maps  # Assuming feature_maps are already in the required shape

    # Step 2: Pass feature maps through MyViT
    output_encoder, indices = encoder_model(feature_maps)
    print(f'Output of Encoder: {output_encoder.shape}')  # Expected: [batch_size, num_samples, hidden_d]
    # Output of Encoder: 2,15,8

