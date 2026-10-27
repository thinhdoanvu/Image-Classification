# predict.py
import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor

# -------------------- Configuration --------------------
input_dir = "/kaggle/input/datasets/thnhonv/catdog/dataset/predict"                                   # ảnh cần dự đoán
output_dir = "inference"                                # kết quả
checkpoint_path = os.path.join("checkpoint", "alexmodel_best.pt")
image_size = 224
image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Phải giống hệt transform lúc train
transform = transforms.Compose([
    Resize((image_size, image_size)),
    ToTensor(),
])


def get_font(img_width):
    """Chọn cỡ chữ theo kích thước ảnh; dùng font có sẵn trên Kaggle/Linux."""
    size = max(14, img_width // 25)
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def draw_label(img, text):
    """Vẽ chữ trên một nền đen bán trong suốt ở góc trên bên trái ảnh."""
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(img.width)

    pad = max(4, img.width // 100)
    left, top, right, bottom = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle(
        (left - pad, top - pad, right + pad, bottom + pad),
        fill=(0, 0, 0, 160),
    )
    draw.text((pad, pad), text, font=font, fill=(255, 255, 0, 255))

    return Image.alpha_composite(img, overlay).convert("RGB")


@torch.no_grad()
def predict_image(model, img, class_names, device):
    x = transform(img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(dim=0)
    return class_names[idx.item()], conf.item(), probs.cpu().tolist()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    from utils.alexnetmodel import Alexnet

    # Load checkpoint đã lưu trong train.py (gồm cả class_names)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = Alexnet(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"Loaded checkpoint (epoch {checkpoint['epoch']}, "
        f"valid acc {checkpoint['valid_accuracy'] * 100:.2f}%)"
    )
    print("Classes:", class_names)

    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(image_exts)
    )
    if not image_files:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong '{input_dir}/'")
    print(f"Found {len(image_files)} images in '{input_dir}/'\n")

    rows = []
    for file_name in image_files:
        img = Image.open(os.path.join(input_dir, file_name)).convert("RGB")

        label, conf, probs = predict_image(model, img, class_names, device)
        print(f"{file_name:<30} -> {label} ({conf * 100:.2f}%)")

        # Vẽ lên ảnh gốc (kích thước gốc), không phải ảnh đã resize
        annotated = draw_label(img, f"{label}: {conf * 100:.2f}%")
        stem = os.path.splitext(file_name)[0]
        annotated.save(os.path.join(output_dir, f"{stem}_pred_{label}.jpg"), quality=95)

        rows.append([file_name, label, f"{conf:.4f}"] + [f"{p:.4f}" for p in probs])

    # Lưu bảng tổng hợp kết quả
    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "predicted_class", "confidence"]
                        + [f"prob_{c}" for c in class_names])
        writer.writerows(rows)

    print(f"\nAnnotated images and predictions.csv saved in: {output_dir}/")
