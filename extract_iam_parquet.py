import os
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

# I originally downloaded the IAM dataset into the IAM-line/data/ folder,
# and it came as train.parquet, validation.parquet, and test.parquet files.
DATASET_DIR = 'IAM-line/data'
SPLITS = ['train', 'validation', 'test']
TROCR_ROOT = 'trocr_data'
OUTPUT_DIRS = {
    'train': os.path.join(TROCR_ROOT, 'train_images'),
    'validation': os.path.join(TROCR_ROOT, 'val_images'),
    'test': os.path.join(TROCR_ROOT, 'test_images'),
}
MAPPING_FILES = {
    'train': os.path.join(TROCR_ROOT, 'train_labels.csv'),
    'validation': os.path.join(TROCR_ROOT, 'val_labels.csv'),
    'test': os.path.join(TROCR_ROOT, 'test_labels.csv'),
}

def extract_split(split):
    parquet_path = os.path.join(DATASET_DIR, f'{split}.parquet')
    out_dir = OUTPUT_DIRS[split]
    mapping_file = MAPPING_FILES[split]
    os.makedirs(out_dir, exist_ok=True)
    
    # Here I'm reading the massive parquet file into memory
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    rows = []
    import io
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Extracting {split} images'):
        img_bytes = row['image']['bytes']
        text = row['text']
        img_name = f'{split}_{idx:05d}.png'
        img_path = os.path.join(out_dir, img_name)
        # The images are stored as raw bytes, so I convert them back to viewable PNG images and save them to disk
        image = Image.open(io.BytesIO(img_bytes))
        image.save(img_path)
        # Generating the mapping row so I can keep track of which image corresponds to which text
        rows.append({'image': img_path, 'text': text})
    # Finally, I save out the whole mapping as a nice, easy-to-read CSV file
    pd.DataFrame(rows).to_csv(mapping_file, index=False)
    print(f'{split.capitalize()} split: {len(rows)} samples extracted.')

def main():
    for split in SPLITS:
        extract_split(split)
    print('All splits extracted and mapping files created.')

if __name__ == '__main__':
    main()
