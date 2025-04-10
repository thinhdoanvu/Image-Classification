import os
import shutil
import random

def splitdata(data_path="../../flowers102", train_ratio=0.8):
    data_path = os.path.abspath(data_path)
    train_dir = "../../train"
    valid_dir = "../../valid"

    # Xóa nếu đã tồn tại
    for split_dir in [train_dir, valid_dir]:
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)

    # Duyệt qua từng lớp
    class_names = [d for d in os.listdir(data_path)
                   if os.path.isdir(os.path.join(data_path, d))
                   and d not in ["train", "valid"]]

    for class_name in class_names:
        src_class_dir = os.path.join(data_path, class_name)
        images = os.listdir(src_class_dir)
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        valid_images = images[split_idx:]

        # Tạo thư mục class trong train/valid
        train_class_dir = os.path.join(train_dir, class_name)
        valid_class_dir = os.path.join(valid_dir, class_name)
        os.makedirs(train_class_dir)
        os.makedirs(valid_class_dir)

        # Move ảnh
        for img in train_images:
            shutil.move(os.path.join(src_class_dir, img), os.path.join(train_class_dir, img))

        for img in valid_images:
            shutil.move(os.path.join(src_class_dir, img), os.path.join(valid_class_dir, img))

    print("✅ Split completed.")
    print(f"Total classes: {len(class_names)}")

splitdata()