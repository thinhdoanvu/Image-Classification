'''
Thinhdv 6May2025
'''
##################################CBAM###########################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x
        return output

class CAM(nn.Module):
    def __init__(self, channels, r=16):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channels, r=16):
        super(CBAMBlock, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x


# Hàm tạo một layer Conv + BN + ReLU
def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# Hàm tạo một Bottleneck Layer (BN → ReLU → Conv 1x1 → BN → ReLU → Conv 3x3 → Dropout)
def bottleneck_layer(in_channels, out_channels, dropout_rate=0.2):
    inter_channels = 4 * out_channels
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(inter_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Dropout(dropout_rate)
    )


# Hàm tạo Transition Layer (BN → ReLU → Conv 1x1 → AvgPool)
def transition_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


# Hàm tạo một Dense Block với vòng lặp
def make_dense_block(in_channels, growth_rate, num_layers, dropout_rate=0.2):
    layers = []
    current_channels = in_channels
    for _ in range(num_layers):
        layer = bottleneck_layer(current_channels, growth_rate, dropout_rate)
        layers.append(layer)
        current_channels += growth_rate
    return nn.Sequential(*layers), current_channels


# DenseNet-121 với CBAM
class DenseNet121_CBAM(nn.Module):
    def __init__(self, num_classes, growth_rate=32, reduction=0.5, dropout_rate=0.2):
        super(DenseNet121_CBAM, self).__init__()

        # Số kênh ban đầu
        num_init_features = 2 * growth_rate

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Block 1 (6 layer)
        self.dense_block1, num_features = make_dense_block(num_init_features, growth_rate, 6, dropout_rate)
        out_features = int(num_features * reduction)
        self.transition1 = transition_layer(num_features, out_features)

        # Dense Block 2 (12 layer) + CBAM
        self.dense_block2, num_features = make_dense_block(out_features, growth_rate, 12, dropout_rate)
        self.cbam2 = CBAMBlock(num_features)
        out_features = int(num_features * reduction)
        self.transition2 = transition_layer(num_features, out_features)

        # Dense Block 3 (24 layer) + CBAM
        self.dense_block3, num_features = make_dense_block(out_features, growth_rate, 24, dropout_rate)
        self.cbam3 = CBAMBlock(num_features)
        out_features = int(num_features * reduction)
        self.transition3 = transition_layer(num_features, out_features)

        # Dense Block 4 (16 layer) + CBAM
        self.dense_block4, num_features = make_dense_block(out_features, growth_rate, 16, dropout_rate)
        self.cbam4 = CBAMBlock(num_features)

        # Final BatchNorm + ReLU
        self.final_bn = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        # Adaptive Avg Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

        # Khởi tạo trọng số
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        # Dense Block 1
        for layer in self.dense_block1:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.transition1(x)

        # Dense Block 2 + HAAM
        for layer in self.dense_block2:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.cbam2(x)
        x = self.transition2(x)

        # Dense Block 3 + HAAM
        for layer in self.dense_block3:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.cbam3(x)
        x = self.transition3(x)

        # Dense Block 4 + HAAM
        for layer in self.dense_block4:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.cbam4(x)
        x = self.final_bn(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


##################################SE###########################################

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


# Hàm tạo một layer Conv + BN + ReLU
def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# Hàm tạo một Bottleneck Layer (BN → ReLU → Conv 1x1 → BN → ReLU → Conv 3x3 → Dropout)
def bottleneck_layer(in_channels, out_channels, dropout_rate=0.2):
    inter_channels = 4 * out_channels
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(inter_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Dropout(dropout_rate)
    )


# Hàm tạo Transition Layer (BN → ReLU → Conv 1x1 → AvgPool)
def transition_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


# Hàm tạo một Dense Block với vòng lặp
def make_dense_block(in_channels, growth_rate, num_layers, dropout_rate=0.2):
    layers = []
    current_channels = in_channels
    for _ in range(num_layers):
        layer = bottleneck_layer(current_channels, growth_rate, dropout_rate)
        layers.append(layer)
        current_channels += growth_rate
    return nn.Sequential(*layers), current_channels


# DenseNet-121 với SE
class DenseNet121_SE(nn.Module):
    def __init__(self, num_classes, growth_rate=32, reduction=0.5, dropout_rate=0.2):
        super(DenseNet121_SE, self).__init__()

        # Số kênh ban đầu
        num_init_features = 2 * growth_rate

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Block 1 (6 layer)
        self.dense_block1, num_features = make_dense_block(num_init_features, growth_rate, 6, dropout_rate)
        out_features = int(num_features * reduction)
        self.transition1 = transition_layer(num_features, out_features)

        # Dense Block 2 (12 layer) + SE
        self.dense_block2, num_features = make_dense_block(out_features, growth_rate, 12, dropout_rate)
        self.se2 = SEBlock(num_features)
        out_features = int(num_features * reduction)
        self.transition2 = transition_layer(num_features, out_features)

        # Dense Block 3 (24 layer) + SE
        self.dense_block3, num_features = make_dense_block(out_features, growth_rate, 24, dropout_rate)
        self.se3 = SEBlock(num_features)
        out_features = int(num_features * reduction)
        self.transition3 = transition_layer(num_features, out_features)

        # Dense Block 4 (16 layer) + SE
        self.dense_block4, num_features = make_dense_block(out_features, growth_rate, 16, dropout_rate)
        self.se4 = SEBlock(num_features)

        # Final BatchNorm + ReLU
        self.final_bn = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        # Adaptive Avg Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

        # Khởi tạo trọng số
        self._initialize_weights()


##################################ECA###########################################
class DenseNet121_SE(nn.Module):
    def __init__(self, num_classes, growth_rate=32, reduction=0.5, dropout_rate=0.2):
        super(DenseNet121_SE, self).__init__()

        num_init_features = 2 * growth_rate

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Block 1
        self.dense_block1, num_features = make_dense_block(num_init_features, growth_rate, 6, dropout_rate)
        out_features = int(num_features * reduction)
        self.transition1 = transition_layer(num_features, out_features)

        # Dense Block 2 + SE
        self.dense_block2, num_features = make_dense_block(out_features, growth_rate, 12, dropout_rate)
        self.se2 = SEBlock(num_features)
        out_features = int(num_features * reduction)
        self.transition2 = transition_layer(num_features, out_features)

        # Dense Block 3 + SE
        self.dense_block3, num_features = make_dense_block(out_features, growth_rate, 24, dropout_rate)
        self.se3 = SEBlock(num_features)
        out_features = int(num_features * reduction)
        self.transition3 = transition_layer(num_features, out_features)

        # Dense Block 4 + SE
        self.dense_block4, num_features = make_dense_block(out_features, growth_rate, 16, dropout_rate)
        self.se4 = SEBlock(num_features)

        # Final BatchNorm + ReLU
        self.final_bn = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        # Adaptive Pool + FC
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        for layer in self.dense_block1:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.transition1(x)

        for layer in self.dense_block2:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.se2(x)
        x = self.transition2(x)

        for layer in self.dense_block3:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.se3(x)
        x = self.transition3(x)

        for layer in self.dense_block4:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.se4(x)

        x = self.final_bn(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



##################################ECA###########################################
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


# Hàm tạo một layer Conv + BN + ReLU
def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# Hàm tạo một Bottleneck Layer (BN → ReLU → Conv 1x1 → BN → ReLU → Conv 3x3 → Dropout)
def bottleneck_layer(in_channels, out_channels, dropout_rate=0.2):
    inter_channels = 4 * out_channels
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(inter_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Dropout(dropout_rate)
    )


# Hàm tạo Transition Layer (BN → ReLU → Conv 1x1 → AvgPool)
def transition_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


# Hàm tạo một Dense Block với vòng lặp
def make_dense_block(in_channels, growth_rate, num_layers, dropout_rate=0.2):
    layers = []
    current_channels = in_channels
    for _ in range(num_layers):
        layer = bottleneck_layer(current_channels, growth_rate, dropout_rate)
        layers.append(layer)
        current_channels += growth_rate
    return nn.Sequential(*layers), current_channels


# DenseNet-121 với ECA
class DenseNet121_ECA(nn.Module):
    def __init__(self, num_classes, growth_rate=32, reduction=0.5, dropout_rate=0.2):
        super(DenseNet121_ECA, self).__init__()

        # Số kênh ban đầu
        num_init_features = 2 * growth_rate

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Block 1 (6 layer)
        self.dense_block1, num_features = make_dense_block(num_init_features, growth_rate, 6, dropout_rate)
        out_features = int(num_features * reduction)
        self.transition1 = transition_layer(num_features, out_features)

        # Dense Block 2 (12 layer) + ECA
        self.dense_block2, num_features = make_dense_block(out_features, growth_rate, 12, dropout_rate)
        self.eca2 = ECA(num_features)
        out_features = int(num_features * reduction)
        self.transition2 = transition_layer(num_features, out_features)

        # Dense Block 3 (24 layer) + ECA
        self.dense_block3, num_features = make_dense_block(out_features, growth_rate, 24, dropout_rate)
        self.eca3 = ECA(num_features)
        out_features = int(num_features * reduction)
        self.transition3 = transition_layer(num_features, out_features)

        # Dense Block 4 (16 layer) + ECA
        self.dense_block4, num_features = make_dense_block(out_features, growth_rate, 16, dropout_rate)
        self.eca4 = ECA(num_features)

        # Final BatchNorm + ReLU
        self.final_bn = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        # Adaptive Avg Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

        # Khởi tạo trọng số
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        # Dense Block 1
        for layer in self.dense_block1:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.transition1(x)

        # Dense Block 2 +ECA
        for layer in self.dense_block2:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.eca2(x)
        x = self.transition2(x)

        # Dense Block 3 + ECA
        for layer in self.dense_block3:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.eca3(x)
        x = self.transition3(x)

        # Dense Block 4 + SE
        for layer in self.dense_block4:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        x = self.eca4(x)
        x = self.final_bn(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
