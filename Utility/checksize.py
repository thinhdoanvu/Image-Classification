import torch
import torch.nn as nn

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class DummyResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, is_first, n_blocks):
        super(DummyResNetLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DummyC2f(nn.Module):
    def __init__(self, channels, use_bn):
        super(DummyC2f, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DummySPPF(nn.Module):
    def __init__(self, in_channels, pool_size):
        super(DummySPPF, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(pool_size)

    def forward(self, x):
        return self.pool(x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define your layers here
        self.resnet_layer1 = DummyResNetLayer(3, 64, 2, True, 1)
        self.resnet_layer2 = DummyResNetLayer(64, 128, 2, False, 3)
        self.c2f_1 = DummyC2f(128, True)
        self.resnet_layer3 = DummyResNetLayer(128, 256, 2, False, 4)
        self.c2f_2 = DummyC2f(256, True)
        self.resnet_layer4 = DummyResNetLayer(256, 512, 2, False, 6)
        self.c2f_3 = DummyC2f(512, True)
        self.resnet_layer5 = DummyResNetLayer(512, 1024, 2, False, 3)
        self.c2f_4 = DummyC2f(1024, True)
        self.sppf = DummySPPF(1024, 5)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(1024, 128, kernel_size=1)
        self.concat = Concat()

    def forward(self, x):
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.c2f_1(x)
        x = self.resnet_layer3(x)
        x = self.c2f_2(x)
        x = self.resnet_layer4(x)
        x = self.c2f_3(x)
        x1 = self.resnet_layer5(x)
        x2 = self.c2f_4(x1)
        x3 = self.sppf(x2)
        print("SPPF Output Size: ", x3.size())
        x4 = self.upsample(x3)
        print("Upsample Output Size: ", x4.size())
        x5 = self.conv(x4)
        print("Conv Output Size: ", x5.size())
        x6 = self.concat([x5, x2])
        print("Concat Output Size: ", x6.size())
        return x6

# Create a dummy input tensor with batch size 1
dummy_input = torch.randn(1, 3, 320, 320)

# Initialize and test the model
model = Model()
output = model(dummy_input)
print("Final Output Size: ", output.size())
