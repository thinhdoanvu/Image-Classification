import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetBackbone(nn.Module):
    def __init__(self, input_size=(244, 244)):
        super(LeNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # đầu vào là ảnh RGB
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Tự động tính toán flatten_dim từ input_size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_size)  # giả lập input
            x = self.pool(F.relu(self.conv1(dummy_input)))  # (1, 6, H, W)
            x = self.pool(F.relu(self.conv2(x)))            # (1, 16, H, W)
            self.flatten_dim = x.view(1, -1).size(1)        # số chiều sau flatten

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 64)  # đầu ra là embedding vector 64 chiều

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 6, H, W)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 16, H, W)
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
