
import os
from torchvision.datasets import ImageFolder

#This function use for subfolders has space character in names
# Define the path to your dataset
dataset_train = '../dataset/train'
dataset_test = '../dataset/test'
dataset_valid = '../dataset/valid'

# Function to replace spaces with underscores in folder names
def preprocess_dataset(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        for dir_name in dirs:
            new_dir_name = dir_name.replace(' ', '_')
            os.rename(os.path.join(root, dir_name), os.path.join(root, new_dir_name))

# Preprocess the dataset to replace spaces with underscores
preprocess_dataset(dataset_train)
preprocess_dataset(dataset_test)
preprocess_dataset(dataset_valid)

