import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# I found out that the TrOCR model specifically expects images to be exactly 384x64 pixels.
# So I made this constant to standardize every single image I feed into it.
TARGET_SIZE = (384, 64)
TARGET_FORMAT = 'PNG'

for split in ['train', 'val', 'test']:
    csv_path = f'trocr_data/{split}_labels.csv'
    df = pd.read_csv(csv_path)
    print(f'Resizing images in {csv_path}...')
    for img_path in tqdm(df['image']):
        try:
            with Image.open(img_path) as im:
                # To keep the math simple and the memory footprint low, I convert everything to grayscale
                im = im.convert('L')
                # Then I squash and stretch the image to my TARGET_SIZE. I use the LANCZOS filter because I read it's the highest quality.
                im = im.resize(TARGET_SIZE, Image.LANCZOS)
                # Finally, I save it back out as a PNG to make sure I don't lose any detail to JPEG compression
                im.save(img_path, format=TARGET_FORMAT)
        except Exception as e:
            print(f'Error processing {img_path}: {e}')
print('All images resized and standardized.')
