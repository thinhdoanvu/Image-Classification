import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import torchvision.models as models
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


def predict():
    clf.eval()

    folder_path = './data/test/'
    # Read the CSV file
    csv_file_path = './data/data.csv'  # path to your CSV file
    df = pd.read_csv(csv_file_path)

    # Get the list of image files in the test folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
    # print(image_files)

    # Iterate over each image file in the test folder
    for image_file in image_files:
        # Process each image file here
        image_path = os.path.join(folder_path, image_file)  # Construct the full path to the image file
        img = Image.open(image_path)
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)

        output = clf(img_tensor)
        predicted_class = torch.argmax(output).item()
        # print(predicted_class)

        # Search for matching rows based on the image file name
        rows = df[df['filepaths'].str.endswith(image_file)]
        if not rows.empty:
            match_found = False
            for index, row in rows.iterrows():
                index_class = row['class index']
                label = row['labels']
                if predicted_class == index_class:
                    print(f"Image: {image_file}, Predicted label: {label}")
                    match_found = True
                    break
            if not match_found:
                print(
                    f"Image: {image_file}, Predicted class index: {predicted_class}, Index class from CSV: {index_class}")
        else:
            print(f"No match found for image: {image_file}")


if __name__ == "__main__":
    device = setup_cuda()

    # 2. Create a segmentation model, then load the trained weights
    from utils.alexnetmodel import ImageClassifier
    num_class = 53
    clf = ImageClassifier(num_class).to(device)
    clf.load_state_dict(torch.load('alexmodel_state.pt', device))
    print('The classify model has been loaded.')

    # 3. Perform segmentation
    predict()
