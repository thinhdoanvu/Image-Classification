# Import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize
import numpy as np

# Get data
train_data = './data/train'

num_class = 53
n_epoch = 500

# Setup CUDA
def setup_cuda():
    # Setting seeds for reproducibility
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
# ----------------DATA----------------------#
class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(train_data, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

transform = transforms.Compose([Resize((224, 224)), ToTensor()])
dataset = PlayingCardDataset(train_data, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
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
    def forward(self, x):
        return self.model(x)



# Instance of the neural network, loss, optimizer
device = setup_cuda()

clf = ImageClassifier().to(device)  # cuda

opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":

    for epoch in range(n_epoch):  # train for 10 epochs
        for batch in train_loader:
            X,y = batch 
            X, y = X.to(device), y.to(device) 
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch:{epoch} loss is {loss.item()}")

        with open('model_state.pt', 'wb') as f:
            save(clf.state_dict(), f)

    # predict
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    img = Image.open('01.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))
    img = Image.open('02.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))
    img = Image.open('03.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))
    img = Image.open('04.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))
    img = Image.open('05.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(clf(img_tensor)))