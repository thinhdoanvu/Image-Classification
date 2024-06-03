"""

Thanh Le  16 April 2024

How to measure the performance of a trained model

"""

import torch

from tqdm import tqdm

from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder

import torchvision.transforms as transforms

from torchvision.transforms import ToTensor, Resize

from torchmetrics.functional import accuracy, precision, recall, f1_score

import numpy as np

from torch import nn, save, load

import pandas as pd

import os



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



def eval_model():

    """

    Evaluate the model over a single epoch

    :return: classification performance, including accuracy, precision, recall, and F1 score

    """

    model.eval()

    acc = 0.0

    pre = 0.0

    rec = 0.0

    f1 = 0.0



    with torch.no_grad():

        for (img, label) in tqdm(test_loader, ncols=80, desc='Evaluation'):

            # Get a batch

            img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)



            # Perform a feed-forward pass

            logits = model(img)



            # Get the predictions

            prediction = logits.argmax(axis=1)



            # Get the predictions to calculate the model's performance for every iteration

            acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=num_classes).item()

            pre += precision(prediction, label, task='multiclass', average='macro', num_classes=num_classes).item()

            rec += recall(prediction, label, task='multiclass', average='macro', num_classes=num_classes).item()

            f1 += f1_score(prediction, label, task='multiclass', average='macro', num_classes=num_classes).item()



    return acc / len(test_loader), pre / len(test_loader), rec / len(test_loader), f1 / len(test_loader)





if __name__ == "__main__":

    device = setup_cuda()



    folder_path = '../dataset/test/'

    # Read the CSV file

    csv_file_path = '../dataset/dataset.csv'  # path to your CSV file



    # Xác định số lớp từ file CSV

    num_classes = get_num_classes_from_csv(csv_file_path)

    print("number of classes:", num_classes)



    # 3. Create a new deep model with pre-trained weights

    from model.cct import cct_14_7x2_224

    model = cct_14_7x2_224(
        num_classes=num_classes,
    ).to(device)




    # 2. Load the weights trained on the Cat-Dog dataset

    folder_checkpoints = 'checkpoints'  # Define the folder name

    file_checkpoints = os.path.join(folder_checkpoints, 'cct_epoch_37_acc_0.7757.pt')  # lay epoch cuoi cung

    model.load_state_dict(torch.load(file_checkpoints, device))

    model.eval()



    # 3. Load the test dataset

    transform = transforms.Compose([Resize((224, 224)), ToTensor()])

    test_dataset = ImageFolder(root=folder_path, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)



    # 4. Evaluate the model on the test set

    acc, pre, rec, f1 = eval_model()

    print(f'Accuracy: {acc}, Precision: {pre}, Recall: {rec}, F1 score: {f1}')

