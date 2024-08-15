import glob
import os
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

files = glob.glob('outputs/camouflaged/*.jpg')
font = ImageFont.truetype('FiraMono-Medium.otf', size=45)

for file in tqdm(files):
    image = Image.open(file)
    W, H = image.size
    draw = ImageDraw.Draw(image)

    draw.text((10, 10), os.path.basename(file)[:-4], fill=(255, 255, 255), font=font)  # text in white color
    image.save(os.path.basename(file))

import os
import moviepy.video.io.ImageSequenceClip  # pip install moviepy
image_folder='temp_images'
image_files = [image_folder + '/' + img for img in os.listdir(image_folder) if img.endswith(".jpg")]
image_files = sorted(image_files, key=lambda s: int(s[18:-4]))  # get from the 5-th letter upto the extension
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=0.5)
clip.write_videofile('video.mp4')
