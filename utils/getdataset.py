#Load image: PIL
#data -> tensor: Dataset + transform
from torch.utils.data import Dataset         #for batch data
from torchvision import transforms           #for 2 tensor, resize, compose
import os                                    #for path
from torchvision.datasets import ImageFolder #for image load
from torchvision.transforms import ToTensor, Resize #resize

train_data = './data/train'
class Getdataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(train_data, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes