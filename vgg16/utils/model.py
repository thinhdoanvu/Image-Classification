import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dropout_percentage = 0.5

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batchnorm1_1 = nn.BatchNorm2d(64)

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm1_2 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm2_1 = nn.BatchNorm2d(128)

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm2_2 = nn.BatchNorm2d(128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batchnorm3_1 = nn.BatchNorm2d(256)
        self.dropout3_1 = nn.Dropout(p=self.dropout_percentage)

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm3_2 = nn.BatchNorm2d(256)
        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage)

        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm3_3 = nn.BatchNorm2d(256)
        self.dropout3_3 = nn.Dropout(p=self.dropout_percentage)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batchnorm4_1 = nn.BatchNorm2d(512)
        self.dropout4_1 = nn.Dropout(p=self.dropout_percentage)

        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm4_2 = nn.BatchNorm2d(512)
        self.dropout4_2 = nn.Dropout(p=self.dropout_percentage)

        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm4_3 = nn.BatchNorm2d(512)
        self.dropout4_3 = nn.Dropout(p=self.dropout_percentage)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm5_1 = nn.BatchNorm2d(512)
        self.dropout5_1 = nn.Dropout(p=self.dropout_percentage)

        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm5_2 = nn.BatchNorm2d(512)
        self.dropout5_2 = nn.Dropout(p=self.dropout_percentage)

        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm5_3 = nn.BatchNorm2d(512)
        self.dropout5_3 = nn.Dropout(p=self.dropout_percentage)

        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.dropout_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Block 1
        x = self.batchnorm1_1(F.relu(self.conv1_1(x)))
        x = self.batchnorm1_2(F.relu(self.conv1_2(x)))
        x = self.maxpool1(x)

        # Block 2
        x = self.batchnorm2_1(F.relu(self.conv2_1(x)))
        x = self.batchnorm2_2(F.relu(self.conv2_2(x)))
        x = self.maxpool2(x)

        # Block 3
        x = self.dropout3_1(self.batchnorm3_1(F.relu(self.conv3_1(x))))
        x = self.dropout3_2(self.batchnorm3_2(F.relu(self.conv3_2(x))))
        x = self.dropout3_3(self.batchnorm3_3(F.relu(self.conv3_3(x))))
        x = self.maxpool3(x)

        # Block 4
        x = self.dropout4_1(self.batchnorm4_1(F.relu(self.conv4_1(x))))
        x = self.dropout4_2(self.batchnorm4_2(F.relu(self.conv4_2(x))))
        x = self.dropout4_3(self.batchnorm4_3(F.relu(self.conv4_3(x))))
        x = self.maxpool4(x)

        # Block 5
        x = self.dropout5_1(self.batchnorm5_1(F.relu(self.conv5_1(x))))
        x = self.dropout5_2(self.batchnorm5_2(F.relu(self.conv5_2(x))))
        x = self.dropout5_3(self.batchnorm5_3(F.relu(self.conv5_3(x))))
        x = self.maxpool5(x)

        # Fully Connected Layers
        x = self.flatten(x)
        x = self.dropout_fc1(F.relu(self.fc1(x)))
        x = self.dropout_fc2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SEVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dropout_percentage = 0.5

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batchnorm1_1 = nn.BatchNorm2d(64)

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm1_2 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se1 = SEBlock(64)


        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm2_1 = nn.BatchNorm2d(128)

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm2_2 = nn.BatchNorm2d(128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se2 = SEBlock(128)


        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batchnorm3_1 = nn.BatchNorm2d(256)
        self.dropout3_1 = nn.Dropout(p=self.dropout_percentage)

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm3_2 = nn.BatchNorm2d(256)
        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage)

        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm3_3 = nn.BatchNorm2d(256)
        self.dropout3_3 = nn.Dropout(p=self.dropout_percentage)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se3 = SEBlock(256)


        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batchnorm4_1 = nn.BatchNorm2d(512)
        self.dropout4_1 = nn.Dropout(p=self.dropout_percentage)

        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm4_2 = nn.BatchNorm2d(512)
        self.dropout4_2 = nn.Dropout(p=self.dropout_percentage)

        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm4_3 = nn.BatchNorm2d(512)
        self.dropout4_3 = nn.Dropout(p=self.dropout_percentage)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se4 = SEBlock(512)


        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm5_1 = nn.BatchNorm2d(512)
        self.dropout5_1 = nn.Dropout(p=self.dropout_percentage)

        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm5_2 = nn.BatchNorm2d(512)
        self.dropout5_2 = nn.Dropout(p=self.dropout_percentage)

        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm5_3 = nn.BatchNorm2d(512)
        self.dropout5_3 = nn.Dropout(p=self.dropout_percentage)

        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se5 = SEBlock(512)


        # Fully Connected
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.dropout_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Block 1
        x = self.batchnorm1_1(F.relu(self.conv1_1(x)))
        x = self.batchnorm1_2(F.relu(self.conv1_2(x)))
        x = self.maxpool1(x)
        x = self.se1(x)


        # Block 2
        x = self.batchnorm2_1(F.relu(self.conv2_1(x)))
        x = self.batchnorm2_2(F.relu(self.conv2_2(x)))
        x = self.maxpool2(x)
        x = self.se2(x)


        # Block 3
        x = self.dropout3_1(self.batchnorm3_1(F.relu(self.conv3_1(x))))
        x = self.dropout3_2(self.batchnorm3_2(F.relu(self.conv3_2(x))))
        x = self.dropout3_3(self.batchnorm3_3(F.relu(self.conv3_3(x))))
        x = self.maxpool3(x)
        x = self.se3(x)


        # Block 4
        x = self.dropout4_1(self.batchnorm4_1(F.relu(self.conv4_1(x))))
        x = self.dropout4_2(self.batchnorm4_2(F.relu(self.conv4_2(x))))
        x = self.dropout4_3(self.batchnorm4_3(F.relu(self.conv4_3(x))))
        x = self.maxpool4(x)
        x = self.se4(x)


        # Block 5
        x = self.dropout5_1(self.batchnorm5_1(F.relu(self.conv5_1(x))))
        x = self.dropout5_2(self.batchnorm5_2(F.relu(self.conv5_2(x))))
        x = self.dropout5_3(self.batchnorm5_3(F.relu(self.conv5_3(x))))
        x = self.maxpool5(x)
        x = self.se5(x)


        # Fully Connected Layers
        x = self.flatten(x)
        x = self.dropout_fc1(F.relu(self.fc1(x)))
        x = self.dropout_fc2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x