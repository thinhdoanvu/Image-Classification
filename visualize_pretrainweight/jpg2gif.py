from PIL import Image, ImageDraw, ImageFont
import glob
import os

# Directory containing JPEG images
image_folder = './output_with_pretrain/'

# Get a list of JPEG image files in the directory and sort them by epoch number
image_files = sorted(glob.glob(os.path.join(image_folder, 'use_pretrained_weights_epoch_*.jpg')),
                     key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

# Create a list to store image objects
frames = []

# Load font
font = ImageFont.truetype("arial.ttf", size=45)

# Define text color
text_color = (255, 255, 255)  # White color

# Open each image, add text, and append to frames list
for image_file in image_files:
    # Open original image
    img = Image.open(image_file)
    width, height = img.size

    # Extract epoch number from filename
    epoch_number = int(os.path.splitext(os.path.basename(image_file))[0].split('_')[-1])

    # Create text content
    text_content = f"Epoch {epoch_number}"

    # Create a new black image with text
    text_img = Image.new('RGB', (200, height), color='black')
    draw = ImageDraw.Draw(text_img)
    draw.text((10, (height - font.getsize(text_content)[1]) / 2), text_content, fill=text_color, font=font)

    # Concatenate text image and original image
    new_img = Image.new('RGB', (width + 200, height))
    new_img.paste(text_img, (0, 0))
    new_img.paste(img, (200, 0))

    # Append concatenated image to frames list
    frames.append(new_img)

# Save frames as a GIF
gif_path = 'output_with_pretrain.gif'
frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
