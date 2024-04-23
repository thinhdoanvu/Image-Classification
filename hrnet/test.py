import os
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms import ToTensor
import torchvision.models as models
import numpy as np
import pandas as pd
import torchvision.transforms
import matplotlib.pyplot as plt

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

def get_num_classes_from_csv(csv_file_path):
    # Đọc file CSV vào DataFrame
    df = pd.read_csv(csv_file_path)

    # Đếm số lớp duy nhất trong cột nhãn
    num_classes = df['labels'].nunique()

    return num_classes

def predict():
    model.eval()

    # Get the list of image files in the test folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
    # print(image_files)

    # Iterate over each image file in the test folder
    for image_file in image_files:
        # Process each image file here
        image_path = os.path.join(folder_path, image_file)  # Construct the full path to the image file
        img = Image.open(image_path)
        plt.imshow(img)
        plt.show()
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)

        output = model(img_tensor)
        predicted_class = torch.argmax(output).item()
        #print(predicted_class)

        # Search for matching rows based on the image file name
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        rows = df[df['filepaths'].str.endswith(image_file)]
        if not rows.empty:
            match_found = False
            for index, row in rows.iterrows():
                index_class = row['class_index']
                label = row['labels']
                if predicted_class == index_class:
                    print(f"Image: {image_file}, Predicted label: {label}")

                    # Draw the predicted label on the image
                    draw = ImageDraw.Draw(img)
                    #Khong co buoc resize de tam anh duoc ro net
                    font = ImageFont.truetype("arial.ttf", 15)
                    text_color = (0, 0, 255)  # blue
                    draw.text((10, 10), f"Predicted label: {label}", (0, 0, 255), font=font)

                    # Save the annotated image with the predicted label as the filename
                    output_folder = 'outputs'
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, f"{label}_{image_file}")
                    img.save(output_path)

                    match_found = True
                    break
            if not match_found:
                print(
                    f"Image: {image_file}, Predicted class index: {predicted_class}, Index class from CSV: {index_class}")
        else:
            print(f"No match found for image: {image_file}")


if __name__ == "__main__":
    device = setup_cuda()

    # Đường dẫn đến file CSV
    folder_path = '../dataset/test/ace_of_clubs/'
    # Read the CSV file
    csv_file_path = '../dataset/dataset.csv'  # path to your CSV file

    # Xác định số lớp từ file CSV
    num_classes = get_num_classes_from_csv(csv_file_path)
    print("number of classes:", num_classes)

    # 3. Create a new deep model with pre-trained weights
    import timm

    # 4. Note that the model pre-trained model has 1,000 output neurons (because ImageNet has 1,000 classes), so we must
    # customize the last linear layer to adapt to our 53-class problem (i.e., Cat vs Dog)
    model = timm.create_model('hrnet_w18', pretrain=True, num_classes=num_classes).to(device)

    folder_checkpoints = 'checkpoints'  # Define the folder name
    file_checkpoints = os.path.join(folder_checkpoints,'hrnet_epoch_78_acc_0.8400.pt')#lay epoch cuoi cung
    model.load_state_dict(torch.load(file_checkpoints, device))

    print('The classify model has been loaded.')

    # 3. Perform segmentation
    predict()
