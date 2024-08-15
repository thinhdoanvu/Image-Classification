import torch
from torch import nn, save, load
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy
from torchvision.transforms import ToTensor, Resize
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Testing for Food-101
train_dir = '../food-101/train'
test_dir = '../food-101/test'
valid_dir = '../food-101/valid'
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32
IMG_SIZE = 224
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
patch_size = 16
# CLASS = {
    # 0: 'apple_pie',
    # 1: 'baby_back_ribs',
    # 2: 'baklava',
    # 3: 'beef_carpaccio',
    # 4: 'beef_tartare',
    # 5: 'beet_salad',
    # 6: 'beignets',
    # 7: 'bibimbap',
    # 8: 'bread_pudding',
    # 9: 'breakfast_burrito',
    # 10: 'bruschetta',
    # 11: 'caesar_salad',
    # 12: 'cannoli',
    # 13: 'caprese_salad',
    # 14: 'carrot_cake',
    # 15: 'ceviche',
    # 16: 'cheesecake',
    # 17: 'cheese_plate',
    # 18: 'chicken_curry',
    # 19: 'chicken_quesadilla',
    # 20: 'chicken_wings',
    # 21: 'chocolate_cake',
    # 22: 'chocolate_mousse',
    # 23: 'churros',
    # 24: 'clam_chowder',
    # 25: 'club_sandwich',
    # 26: 'crab_cakes',
    # 27: 'creme_brulee',
    # 28: 'croque_madame',
    # 29: 'cup_cakes',
    # 30: 'deviled_eggs',
    # 31: 'donuts',
    # 32: 'dumplings',
    # 33: 'edamame',
    # 34: 'eggs_benedict',
    # 35: 'escargots',
    # 36: 'falafel',
    # 37: 'filet_mignon',
    # 38: 'fish_and_chips',
    # 39: 'foie_gras',
    # 40: 'french_fries',
    # 41: 'french_onion_soup',
    # 42: 'french_toast',
    # 43: 'fried_calamari',
    # 44: 'fried_rice',
    # 45: 'frozen_yogurt',
    # 46: 'garlic_bread',
    # 47: 'gnocchi',
    # 48: 'greek_salad',
    # 49: 'grilled_cheese_sandwich',
    # 50: 'grilled_salmon',
    # 51: 'guacamole',
    # 52: 'gyoza',
    # 53: 'hamburger',
    # 54: 'hot_and_sour_soup',
    # 55: 'hot_dog',
    # 56: 'huevos_rancheros',
    # 57: 'hummus',
    # 58: 'ice_cream',
    # 59: 'lasagna',
    # 60: 'lobster_bisque',
    # 61: 'lobster_roll_sandwich',
    # 62: 'macaroni_and_cheese',
    # 63: 'macarons',
    # 64: 'miso_soup',
    # 65: 'mussels',
    # 66: 'nachos',
    # 67: 'omelette',
    # 68: 'onion_rings',
    # 69: 'oysters',
    # 70: 'pad_thai',
    # 71: 'paella',
    # 72: 'pancakes',
    # 73: 'panna_cotta',
    # 74: 'peking_duck',
    # 75: 'pho',
    # 76: 'pizza',
    # 77: 'pork_chop',
    # 78: 'poutine',
    # 79: 'prime_rib',
    # 80: 'pulled_pork_sandwich',
    # 81: 'ramen',
    # 82: 'ravioli',
    # 83: 'red_velvet_cake',
    # 84: 'risotto',
    # 85: 'samosa',
    # 86: 'sashimi',
    # 87: 'scallops',
    # 88: 'seaweed_salad',
    # 89: 'shrimp_and_grits',
    # 90: 'spaghetti_bolognese',
    # 91: 'spaghetti_carbonara',
    # 92: 'spring_rolls',
    # 93: 'steak',
    # 94: 'strawberry_shortcake',
    # 95: 'sushi',
    # 96: 'tacos',
    # 97: 'takoyaki',
    # 98: 'tiramisu',
    # 99: 'tuna_tartare',
    # 100: 'waffles',
# }  # TO BE CONTINUE


# GPU or CPU
def setup_cuda():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


device = setup_cuda()

# transform images
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# Prediction
def predict_image(image_path, model, transform, class_names, device):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
    predicted_label = class_names[predicted_class.item()]
    return img, predicted_label


# Test function
def test_model():
    # 1. Tải dữ liệu và lớp từ tập train
    transform = transforms.Compose([Resize((224, 224)), ToTensor()])
    train_dataset = ImageFolder(root='../data/train', transform=transform)
    test_dataset = ImageFolder(root='../data/test', transform=transform)
    # Get class names
    class_names = train_dataset.classes

    # 2. Load model
    from utils.coordinateresnet18 import CAResNet18

    model = CAResNet18(
        num_classes=len(class_names),
    ).to(device)

    folder_checkpoint = 'checkpoints_se_resnet18'  # Define the folder name
    file_name = 'se_resnet18_epoch_7_acc_0.8208.pt' # best weight
    file_checkpoint = os.path.join(folder_checkpoint, file_name)  # lay best weight
    model.load_state_dict(torch.load(file_checkpoint, device))
    print('Model loaded from checkpoint.')
    # Ensure the output directory exists
    output_dir = "output_resnet18"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Predict on test
    for image_path in tqdm(test_dataset.imgs, desc='Testing'):
        img, predicted_label = predict_image(image_path[0], model, manual_transforms, class_names, device)
        # plt.imshow(img)
        # plt.title(f'Predicted: {predicted_label}')
        # plt.show()

        # Convert the tensor image back to a PIL image if necessary
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        # Create a plot
        fig, ax = plt.subplots()

        # Set white background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Remove axis
        ax.axis('off')

        # Display the image
        ax.imshow(img)

        # Add the predicted label as the title
        ax.set_title(f'Predicted: {predicted_label}', fontsize=12, pad=10)

        # Save the figure
        image_basename = os.path.basename(image_path[0])
        image_name, image_ext = os.path.splitext(image_basename)
        output_image_path = os.path.join(output_dir, f"{image_name}_pred_{predicted_label}.png")

        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


if __name__ == '__main__':
    test_model()
