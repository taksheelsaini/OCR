import pandas as pd
import os
from sklearn.model_selection import train_test_split

# First, I need to gather all the samples from the original splits so I can re-shuffle them fairly
all_samples = []
for split in ['train', 'val', 'test']:
    csv_path = f'trocr_data/{split}_labels.csv'
    df = pd.read_csv(csv_path)
    all_samples.append(df)
df_all = pd.concat(all_samples, ignore_index=True)

# Now I shuffle everything and split it into my final 80/10/10 ratio for Training/Validation/Testing
train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# Finally, I save these new perfect splits back out to the trocr_data folder
train_df.to_csv('trocr_data/final_train_labels.csv', index=False)
val_df.to_csv('trocr_data/final_val_labels.csv', index=False)
test_df.to_csv('trocr_data/final_test_labels.csv', index=False)

print(f"Final splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.")
