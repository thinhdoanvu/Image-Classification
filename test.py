import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    device = setup_cuda()

    # Load the model
    clf = models.resnet18(pretrained=True).to(device)
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(torch.load(f))
    clf.eval()  # Set the model to evaluation mode

    # Define the device (CPU or GPU) to run the model on
    clf.to(device)
    folder_path = './data/test'
    # Read the CSV file
    csv_file_path = './data/data.csv'  #path to your CSV file
    df = pd.read_csv(csv_file_path)

    # Get the list of subfolders (each representing a class)
    class_folders = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]
    # print(class_folders)

    # Iterate over each class folder
    for class_folder in class_folders:
        # Get the path to the current class folder
        class_folder_path = os.path.join(folder_path, class_folder)
        print(class_folder_path)
        # Get the list of image files in the class folder
        image_files = [file for file in os.listdir(class_folder_path) if file.endswith('.jpg')]
        print(image_files)

        # Iterate over each image file in the class folder
        for image_file in image_files:
            # Process each image file here
            # Predict for each image
            image_path = os.path.join(class_folder_path, image_file)
            img = Image.open(image_path)
            img_tensor = ToTensor()(img).unsqueeze(0).to(device)

            # Perform inference
            with torch.no_grad():
                output = clf(img_tensor)
                predicted_class = torch.argmax(output).item()

            row = df[df['filepaths'].str.endswith(image_file)]
            if not row.empty:
                # Get the index class from the matched row
                index_class = row['labels'].values[0]
                print(f"Image: {image_file}, Predicted class index: {index_class}")
            else:
                print(f"No match found for image: {image_file}")
