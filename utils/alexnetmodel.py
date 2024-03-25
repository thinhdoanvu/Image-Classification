from torch import nn

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.model = nn.Sequential(
            #### Convolutional Layers ####
            #input: 224*224*3
            # Layer 1:
            nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4)),  # Change 1 to 3 for RGB images: output = 32
            #output: W = (224-11+2*0)/4 + 1=54, H = (224-11+2*0)/4 + 1=54
            nn.ReLU(),  # output:96, W=H=54 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(96),
            nn.MaxPool2d(2, 2),
            # after pooling: ((W=H=54-kernel=2)/stride=2) + 1= 27; nn.Flatten(), nn.Linear(96*(27)*(27), 53),

            # Layer 2
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=(2, 2)),  # input: 96, output: 256
            #output: W=(27 - 5 + 2 * 2) / 1 + 1 = 26, H=(27 - 5 + 2 * 2) / 1 + 1 = 26
            nn.ReLU(),  # output:256, W=H=26 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2),
            # after pooling: ((W=H=26-kernel=3)/stride=2)+1 = 12; nn.Flatten(), nn.Linear(256*(12)*(12), 53),

            # Layer 3
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),  # input: 256, output: 384
            # output: W=(12 - 3 + 2 * 1) / 1 + 1 = 12, W=(12 - 3 + 2 * 1) / 1 + 1 = 12
            nn.ReLU(),  # output:384, W=H=12 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(384),

            # Layer 4
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=(1, 1)),  # input: 384, output: 384
            # output: W=(12 - 3 + 2 * 1) / 1 + 1 = 12, W=(12 - 3 + 2 * 1) / 1 + 1 = 12
            nn.ReLU(),  # output:384, W=H=12 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(384),

            # Layer 5
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),  # input: 384, output: 256
            # output: W=(12 - 3 + 2 * 1) / 1 + 1 = 12, W=(12 - 3 + 2 * 1) / 1 + 1 = 12
            nn.ReLU(),  # output:256, W=H=12 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2),
            # after pooling: ((W=H=12-kernel=3)/stride=2)+1 = 6; nn.Flatten(), nn.Linear(256*(6)*(6), 53),

            #### Fully-Connected Layer ####
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256*6 *6, num_class), #tinh sao ra bang 6??????
        )
        #Truyen tham so vao
        self.num_class = num_class

    def forward(self, x):
        return self.model(x)
