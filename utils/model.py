from torch import nn

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.model = nn.Sequential(
            #### Convolutional Layers ####
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=(3, 3)),  # Change 1 to 3 for RGB images: output = 32
            nn.ReLU(),  # output:32, W=H=224px -2 chieu dai moi canh sau conv = 222
            nn.MaxPool2d(2, 2),  # after pooling: W=H=222/2 = 111; nn.Flatten(), nn.Linear(32*(111)*(111), 53),
            nn.Dropout(0.2),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),  # input: 32, output: 64
            nn.ReLU(),  # output:64, W=H=111px do co cung chieu dai moi canh sau conv
            nn.Conv2d(64, 64, kernel_size=(3, 3)),  # after conv: W=H=(111-2)/2 = 54.5; lay phan nguyen = 54
            nn.MaxPool2d(2, 2),  # after pooling: 54; nn.Flatten(), nn.Linear(64*(54)*(54), 53),
            nn.Dropout(0.2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),  # input: 64, output: 128
            nn.ReLU(),  # output:128, W=H=54px do co cung chieu dai moi canh sau conv
            nn.Conv2d(128, 128, kernel_size=(3, 3)),  # after conv: W=H=(54-2) = 52
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # after pooling: (52)/2 = 26; nn.Flatten(), nn.Linear(128*(26)*(26), 53),
            nn.Dropout(0.2),

            # Layer 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),  # input: 128, output: 256
            nn.ReLU(),  # output:256, W=H=26px do co cung chieu dai moi canh sau conv
            nn.Conv2d(256, 256, kernel_size=(3, 3)),  # after conv: W=H=(26-2) = 24
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # after pooling: (24)/2 = 12; nn.Flatten(), nn.Linear(256*(12)*(12), 53),
            nn.Dropout(0.2),

            # Layer 5
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),  # input: 256, output: 512
            nn.ReLU(),  # output:512, W=H=12px do co cung chieu dai moi canh sau conv
            nn.Conv2d(512, 512, kernel_size=(3, 3)),  # after conv: W=H=(12-2) = 10
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # after pooling: (10)/2 = 5; nn.Flatten(), nn.Linear(512*(5)*(5), 53),
            nn.Dropout(0.2),

            #### Fully-Connected Layer ####
            nn.Flatten(), nn.Linear(512 * (5) * (5), num_class),
        )
        #Truyen tham so vao
        self.num_class = num_class

    def forward(self, x):
        return self.model(x)
