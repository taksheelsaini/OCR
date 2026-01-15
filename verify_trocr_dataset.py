import os
import pandas as pd
from PIL import Image

def verify_mapping(mapping_csv):
    df = pd.read_csv(mapping_csv)
    errors = []
    for i, row in df.iterrows():
        img_path = row['image']
        text = str(row['text'])
        # First, I check if the image file actually exists on my hard drive
        if not os.path.exists(img_path):
            errors.append(f"Missing image: {img_path}")
            continue
        # Next, I try to open the image to make sure it isn't corrupted or empty
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            errors.append(f"Corrupted image: {img_path} ({e})")
        # Finally, I double-check that the text label actually contains real data and isn't just blank
        if not text or text.strip() == '' or text.lower() == 'nan':
            errors.append(f"Empty or missing text for image: {img_path}")
    print(f"Checked {len(df)} samples in {mapping_csv}")
    if errors:
        print(f"Found {len(errors)} issues:")
        for err in errors:
            print(err)
    else:
        print("No issues found!")

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        mapping = f"trocr_data/{split}_labels.csv"
        print(f"\nVerifying {mapping}...")
        verify_mapping(mapping)
