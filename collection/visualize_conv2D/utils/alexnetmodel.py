import torch
import torch.nn as nn

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        # Layer 1:
        self.con1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(96)
        self.max1 = nn.MaxPool2d(2, 2)  # Pooling after ReLU

        # Layer 2:
        self.con2 = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=(2, 2))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(256)
        self.max2 = nn.MaxPool2d(3, 2)

        # Layer 3:
        self.con3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(384)

        # Layer 4:
        self.con4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(384)

        # Layer 5:
        self.con5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(256)
        self.max5 = nn.MaxPool2d(3, 2)

        # Fully-Connected Layer:
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * 6 * 6, num_classes)  # Assuming final feature map size is 6x6

    def forward(self, x):
        op1 = self.max1(self.bn1(self.relu1(self.con1(x))))
        op2 = self.max2(self.bn2(self.relu2(self.con2(op1))))
        op3 = self.bn3(self.relu3(self.con3(op2)))
        op4 = self.bn4(self.relu4(self.con4(op3)))
        op5 = self.max5(self.bn5(self.relu5(self.con5(op4))))

        # Flatten and apply fully connected layer
        flattened = self.flatten(op5)
        dropped = self.dropout(flattened)
        out = self.fc(dropped)

        return [op1, op2, op3, op4, op5]
