import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# VGG19
def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         conv_block(64, 64),
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         conv_block(128, 128),
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(),
                                         *[conv_block(256, 256) for _ in range(3)],
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(),
                                         *[conv_block(512, 512) for _ in range(3)],
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block5 = nn.Sequential(*[conv_block(512, 512) for _ in range(4)],
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7,7))

        self.linear1 = nn.Sequential(nn.Linear(in_features=7*7*512, out_features=4096, bias=True),
                                     nn.Dropout(0.5),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(in_features=4096, out_features=4096, bias=True),
                                     nn.Dropout(0.5),
                                     nn.ReLU())
        self.linear3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.avg_pool(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.linear2(x)
        x = self.linear3(x)
        return x


# SE + VGG16


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEVGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         conv_block(64, 64),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         SEBlock(64))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(128))

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            *[conv_block(256, 256) for _ in range(3)],
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(256))

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            *[conv_block(512, 512) for _ in range(3)],
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(512))

        self.conv_block5 = nn.Sequential(*[conv_block(512, 512) for _ in range(4)],
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         SEBlock(512))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.linear1 = nn.Sequential(nn.Linear(in_features=7 * 7 * 512, out_features=4096, bias=True),
                                     nn.Dropout(0.5),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(in_features=4096, out_features=4096, bias=True),
                                     nn.Dropout(0.5),
                                     nn.ReLU())
        self.linear3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.avg_pool(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.linear2(x)
        x = self.linear3(x)
        return x



# CBAM + VGG16

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
                              bias=self.bias)

    def forward(self, x):
        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max, avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x
        return output


class CAM(nn.Module):
    def __init__(self, channels, r=16):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels // self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b, c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output


class CBAM(nn.Module):
    def __init__(self, channels, r=16):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x


class CBAMVGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         conv_block(64, 64),
                                         CBAM(64, r=2),
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            conv_block(128, 128),
            CBAM(128, r=2),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            *[conv_block(256, 256) for _ in range(3)],
            CBAM(256, r=2),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            *[conv_block(512, 512) for _ in range(3)],
            CBAM(512, r=2),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block5 = nn.Sequential(*[conv_block(512, 512) for _ in range(4)],
                                         CBAM(512, r=2),
                                         nn.MaxPool2d(kernel_size=2, stride=2))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.linear1 = nn.Sequential(nn.Linear(in_features=7 * 7 * 512, out_features=4096, bias=True),
                                     nn.Dropout(0.5),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(in_features=4096, out_features=4096, bias=True),
                                     nn.Dropout(0.5),
                                     nn.ReLU())
        self.linear3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.avg_pool(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.linear2(x)
        x = self.linear3(x)
        return x


# ECA + VGG16

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()

        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1  # Đảm bảo kernel size luôn là số lẻ

        # Khởi tạo các layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]

        # Đưa về dạng [B, 1, C] để conv1d
        y = y.squeeze(-1).transpose(-1, -2)  # [B, C, 1] => [B, 1, C]
        y = self.conv1d(y)  # [B, 1, C]

        # Đưa về lại [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, 1, C] => [B, C, 1] => [B, C, 1, 1]
        y = self.sigmoid(y)  # Trọng số chú ý [B, C, 1, 1]

        # Nhân trọng số kênh với đầu vào
        return x * y.expand_as(x)


class ECAVGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         conv_block(64, 64),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         ECA(64))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ECA(128))

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            *[conv_block(256, 256) for _ in range(3)],
            nn.MaxPool2d(kernel_size=2, stride=2),
            ECA(256))

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            *[conv_block(512, 512) for _ in range(3)],
            nn.MaxPool2d(kernel_size=2, stride=2),
            ECA(512))

        self.conv_block5 = nn.Sequential(*[conv_block(512, 512) for _ in range(4)],
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         ECA(512))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.linear1 = nn.Sequential(nn.Linear(in_features=7 * 7 * 512, out_features=4096, bias=True),
                                     nn.Dropout(0.5),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(in_features=4096, out_features=4096, bias=True),
                                     nn.Dropout(0.5),
                                     nn.ReLU())
        self.linear3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.avg_pool(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.linear2(x)
        x = self.linear3(x)
        return x