import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn


class ConvStem(nn.Module):  # input 3x224x224, output 256x28x28
    def __init__(self, c1=3, c2=256):  # c1: input chanel, c2: output chanel
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1:
            nn.Conv2d(in_channels=c1, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            # Change 1 to 3 for RGB images: output = 32
            # output: W = (224-7+2*1)/2 + 1=112, H = (224-11+2*1)/2 + 1=112
            nn.ReLU(),  # output:96, W=H=112 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(96),
            nn.MaxPool2d(2, 2),
            # after pooling: ((W=H=112-kernel=2)/stride=2) + 1= 56; Neu nn.Flatten(), nn.Linear(96*(56)*(56), 53),

            # Layer 2
            nn.Conv2d(in_channels=96, out_channels=c2, kernel_size=(5, 5), padding=(2, 2)),  # input: 96, output: 256
            # output: W=(56 - 5 + 2 * 2) / 1 + 1 = 56, H=(56 - 5 + 2 * 2) / 1 + 1 = 56
            nn.ReLU(),  # output:256, W=H=56 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(c2),
            nn.MaxPool2d(3, 2, padding=1),
            # after pooling: ((W=H=56-kernel=3)/stride=2)+1 = 28; Neu nn.Flatten(), nn.Linear(256*(28)*(28), 53),
        )

    def forward(self, x):
        x = self.model(x)
        return x
