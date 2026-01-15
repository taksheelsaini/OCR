import pandas as pd
import re
import os
import unicodedata
from tqdm import tqdm

def normalize_text(text):
    """
    I wrote this helper function to clean up the messy text labels.
    """
    # First, I apply Unicode normalization so weirdly encoded characters don't break my model
    text = unicodedata.normalize('NFKC', str(text))
    # Then I use a regex to strip out any bizarre symbols, but I keep basic punctuation, letters, and numbers
    text = re.sub(r"[^\w\s.,'\"!?;:\-()\[\]]", '', text)
    # Sometimes there are totally random double or triple spaces, so I collapse those down into single spaces
    text = re.sub(r'\s+', ' ', text)
    # Finally, I strip any accidental leading or trailing whitespace so everything is perfectly clean
    text = text.strip()
    return text

def normalize_labels(csv_path):
    df = pd.read_csv(csv_path)
    orig_texts = df['text'].tolist()
    # Here I loop through every single text label and apply my normalization function
    df['text'] = [normalize_text(t) for t in tqdm(orig_texts, desc=f'Normalizing {os.path.basename(csv_path)}')]
    
    # I experimented with spell correction here, but decided to comment it out for now.
    # It's an interesting idea I might revisit if I need the ground truth to be mathematically perfect!
    # import spellchecker
    # spell = spellchecker.SpellChecker()
    # df['text'] = [' '.join([spell.correction(w) for w in t.split()]) for t in df['text']]
    df.to_csv(csv_path, index=False)
    print(f'Normalized {csv_path}')

if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        csv_path = f'trocr_data/{split}_labels.csv'
        normalize_labels(csv_path)
