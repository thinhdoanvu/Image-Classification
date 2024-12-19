import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
# src_folder = 'Negative'
# train_folder = 'Negative/train'
# valid_folder = 'Negative/valid'
# test_folder = 'Negative/test'

src_folder = 'Positive'
train_folder = 'Positive/train'
valid_folder = 'Positive/valid'
test_folder = 'Positive/test'

# Create directories if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get list of image files
all_images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

# Split data into train and temp (valid + test)
train_images, temp_images = train_test_split(all_images, test_size=0.3, random_state=42)

# Split temp into valid and test
valid_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

# Move images to their respective folders
def move_files(file_list, src_folder, dst_folder):
    for file_name in file_list:
        src_file = os.path.join(src_folder, file_name)
        dst_file = os.path.join(dst_folder, file_name)
        shutil.copy(src_file, dst_file)

# Move the files
move_files(train_images, src_folder, train_folder)
move_files(valid_images, src_folder, valid_folder)
move_files(test_images, src_folder, test_folder)

print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(valid_images)} images")
print(f"Test set: {len(test_images)} images")
