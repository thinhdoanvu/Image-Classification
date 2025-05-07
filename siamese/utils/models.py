import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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


class ResNet18Backbone(nn.Module):
    def __init__(self, embedding_dim=64, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)

        # Bỏ classifier cuối cùng, giữ lại phần feature extractor
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # bỏ fc cuối
        self.fc = nn.Linear(resnet.fc.in_features, embedding_dim)  # chuyển sang vector 64 chiều

    def forward(self, x):
        x = self.feature_extractor(x)  # output shape: (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # flatten: (B, 512)
        x = self.fc(x)  # embedding: (B, embedding_dim)
        return x



class VGG16Backbone(nn.Module):
    def __init__(self, embedding_dim=64):
        super(VGG16Backbone, self).__init__()
        # Tải mô hình VGG16 pretrained
        vgg16 = models.vgg16(pretrained=True)

        # Sử dụng phần feature extractor (bỏ phần classifier gốc)
        self.features = vgg16.features  # gồm các conv + pooling

        # Đóng băng trọng số nếu bạn muốn chỉ huấn luyện phần classifier cuối
        # for param in self.features.parameters():
        #     param.requires_grad = False

        # Tính đầu ra flatten tự động
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy_input)
            self.flatten_dim = out.view(1, -1).size(1)

        # Embedding layers (tùy chỉnh cho Siamese)
        self.embedding = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x


class VGG16Backbone_v2(nn.Module):
    def __init__(self, num_classes=102):
        super(VGG16Backbone_v2, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        self.block1 = nn.Sequential(*vgg16.features[:5])  # out: [B, 64, 112, 112]
        self.block2 = nn.Sequential(*vgg16.features[5:10])  # out: [B, 128, 56, 56]
        self.block3 = nn.Sequential(*vgg16.features[10:17])  # out: [B, 256, 28, 28]
        self.block4 = nn.Sequential(*vgg16.features[17:24])  # out: [B, 512, 14, 14]
        self.block5 = nn.Sequential(*vgg16.features[24:31])  # out: [B, 512, 7, 7]

        # Average pooling + FC cho output chính
        self.avgpool = vgg16.avgpool
        self.fc = nn.Sequential(*vgg16.classifier[:4])  # FC1 + ReLU + Dropout + FC2
        self.head = nn.Linear(4096, num_classes)  # FC3: phân loại chính

        # Các auxiliary heads (sử dụng AdaptiveAvgPool2d + Linear)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, num_classes)  # block1
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, num_classes)  # block2
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, num_classes)  # block3
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, num_classes)  # block4
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, num_classes)  # block5
            ),
        ])

    def forward(self, x):
        aux_outputs = []

        x = self.block1(x)
        aux_outputs.append(self.aux_heads[0](x))

        x = self.block2(x)
        aux_outputs.append(self.aux_heads[1](x))

        x = self.block3(x)
        aux_outputs.append(self.aux_heads[2](x))

        x = self.block4(x)
        aux_outputs.append(self.aux_heads[3](x))

        x = self.block5(x)
        aux_outputs.append(self.aux_heads[4](x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        main_logits = self.head(x)

        return main_logits, aux_outputs


