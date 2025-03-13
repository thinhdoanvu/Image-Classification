import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG16
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

# SE + VGG16

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

# CBAM + VGG16
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


class CBAMVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dropout_percentage = 0.5

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batchnorm1_1 = nn.BatchNorm2d(64)

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm1_2 = nn.BatchNorm2d(64)
        
        self.cbam1 = CBAM(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm2_1 = nn.BatchNorm2d(128)

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm2_2 = nn.BatchNorm2d(128)

        self.cbam2 = CBAM(128)
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

        self.cbam3 = CBAM(256)
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

        self.cbam4 = CBAM(512)
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

        self.cbam5 = CBAM(512)
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
        x = self.cbam1(x)
        x = self.maxpool1(x)


        # Block 2
        x = self.batchnorm2_1(F.relu(self.conv2_1(x)))
        x = self.batchnorm2_2(F.relu(self.conv2_2(x)))
        x = self.cbam2(x)
        x = self.maxpool2(x)
        

        # Block 3
        x = self.dropout3_1(self.batchnorm3_1(F.relu(self.conv3_1(x))))
        x = self.dropout3_2(self.batchnorm3_2(F.relu(self.conv3_2(x))))
        x = self.dropout3_3(self.batchnorm3_3(F.relu(self.conv3_3(x))))
        x = self.cbam3(x)
        x = self.maxpool3(x)
        

        # Block 4
        x = self.dropout4_1(self.batchnorm4_1(F.relu(self.conv4_1(x))))
        x = self.dropout4_2(self.batchnorm4_2(F.relu(self.conv4_2(x))))
        x = self.dropout4_3(self.batchnorm4_3(F.relu(self.conv4_3(x))))
        x = self.cbam4(x)
        x = self.maxpool4(x)
        

        # Block 5
        x = self.dropout5_1(self.batchnorm5_1(F.relu(self.conv5_1(x))))
        x = self.dropout5_2(self.batchnorm5_2(F.relu(self.conv5_2(x))))
        x = self.dropout5_3(self.batchnorm5_3(F.relu(self.conv5_3(x))))
        x = self.cbam5(x)
        x = self.maxpool5(x)
        

        # Fully Connected
        x = self.flatten(x)
        x = self.dropout_fc1(F.relu(self.fc1(x)))
        x = self.dropout_fc2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


# ECA + VGG16

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()

        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1  # Đảm bảo kernel size luôn là số lẻ
        
        # Khởi tạo các layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False) 
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
        

class ECAVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dropout_percentage = 0.5

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batchnorm1_1 = nn.BatchNorm2d(64)

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm1_2 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.eca1 = ECA(64)
        

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm2_1 = nn.BatchNorm2d(128)

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm2_2 = nn.BatchNorm2d(128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.eca2 = ECA(128)
        

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
        self.eca3 = ECA(256)
        

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
        self.eca4 = ECA(512)
        

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
        self.eca5 = ECA(512)
        

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
        op1 = self.eca1(x)
        

        # Block 2
        x = self.batchnorm2_1(F.relu(self.conv2_1(op1)))
        x = self.batchnorm2_2(F.relu(self.conv2_2(x)))
        x = self.maxpool2(x)
        op2 = self.eca2(x)
        

        # Block 3
        x = self.dropout3_1(self.batchnorm3_1(F.relu(self.conv3_1(op2))))
        x = self.dropout3_2(self.batchnorm3_2(F.relu(self.conv3_2(x))))
        x = self.dropout3_3(self.batchnorm3_3(F.relu(self.conv3_3(x))))
        x = self.maxpool3(x)
        op3 = self.eca3(x)
        

        # Block 4
        x = self.dropout4_1(self.batchnorm4_1(F.relu(self.conv4_1(op3))))
        x = self.dropout4_2(self.batchnorm4_2(F.relu(self.conv4_2(x))))
        x = self.dropout4_3(self.batchnorm4_3(F.relu(self.conv4_3(x))))
        x = self.maxpool4(x)
        op4 = self.eca4(x)
        

        # Block 5
        x = self.dropout5_1(self.batchnorm5_1(F.relu(self.conv5_1(op4))))
        x = self.dropout5_2(self.batchnorm5_2(F.relu(self.conv5_2(x))))
        x = self.dropout5_3(self.batchnorm5_3(F.relu(self.conv5_3(x))))
        x = self.maxpool5(x)
        op5 = self.eca5(x)
        

        # Fully Connected Layers
        x = self.flatten(op5)
        x = self.dropout_fc1(F.relu(self.fc1(x)))
        x = self.dropout_fc2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
