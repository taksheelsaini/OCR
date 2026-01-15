"""
Step 16: My Ultimate Benchmark - Fine-tuned TrOCR vs Baseline TrOCR vs EasyOCR
I built this script to formally compare my models. I designed it to load and test ONE MODEL AT A TIME 
because my computer's CPU was running out of memory when I tried to load them all at once!
"""

import os
import sys
import gc
import json
import pandas as pd
from PIL import Image
import torch
import random

sys.stdout.reconfigure(encoding='utf-8')

# ========== MY CONFIGURATION SETTINGS ==========
# These are the paths to the models I'm comparing
FINETUNED_DIR = 'trocr_finetuned'
BASELINE_NAME = 'microsoft/trocr-base-handwritten'
TEST_CSV = 'trocr_data/final_test_labels.csv'
NUM_TEST_SAMPLES = 20
RESULTS_FILE = 'comparison_results.json'

# ========== MY CER CALCULATION FUNCTION ==========
# I wrote this dynamic programming function to calculate Character Error Rate without any external libraries.
def calculate_cer(pred, truth):
    if len(truth) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    m, n = len(pred), len(truth)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == truth[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n] / len(truth)

# ========== SELECTING MY TEST SAMPLES ==========
# I'll just load my test set CSV using pandas
test_df = pd.read_csv(TEST_CSV)
random.seed(42)
indices = random.sample(range(len(test_df)), min(NUM_TEST_SAMPLES, len(test_df)))
test_subset = test_df.iloc[indices].reset_index(drop=True)

# I need to verify all images exist before testing, otherwise the script might crash halfway through
valid_rows = []
for _, row in test_subset.iterrows():
    if os.path.exists(row['image']):
        valid_rows.append({'image': row['image'], 'text': str(row['text'])})
print(f"Testing on {len(valid_rows)} valid images")

print("=" * 70)
print("STEP 16: 3-WAY MODEL COMPARISON")
print("Fine-tuned TrOCR  vs  Baseline TrOCR  vs  EasyOCR")
print("=" * 70)

# ========== TEST 1: MY FINE-TUNED TrOCR ==========
# First up is the model I trained myself!
print("\n--- [1/3] Testing MY FINE-TUNED TrOCR ---")
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ft_processor = TrOCRProcessor.from_pretrained(FINETUNED_DIR)
ft_model = VisionEncoderDecoderModel.from_pretrained(FINETUNED_DIR)
ft_model.eval()

ft_predictions = []
for i, row in enumerate(valid_rows):
    image = Image.open(row['image']).convert('RGB')
    pixel_values = ft_processor(image, return_tensors='pt').pixel_values
    with torch.no_grad():
        ids = ft_model.generate(pixel_values, max_length=64, num_beams=4)
    pred = ft_processor.batch_decode(ids, skip_special_tokens=True)[0]
    ft_predictions.append(pred)
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{len(valid_rows)} done")

del ft_model, ft_processor
gc.collect()
print("  Fine-tuned model unloaded from memory")

# ========== TEST 2: THE BASELINE TrOCR ==========
# Now I load the original Microsoft model to see how much I improved it
print("\n--- [2/3] Testing BASELINE TrOCR ---")
bl_processor = TrOCRProcessor.from_pretrained(BASELINE_NAME)
bl_model = VisionEncoderDecoderModel.from_pretrained(BASELINE_NAME)
bl_model.eval()

bl_predictions = []
for i, row in enumerate(valid_rows):
    image = Image.open(row['image']).convert('RGB')
    pixel_values = bl_processor(image, return_tensors='pt').pixel_values
    with torch.no_grad():
        ids = bl_model.generate(pixel_values, max_length=64, num_beams=4)
    pred = bl_processor.batch_decode(ids, skip_special_tokens=True)[0]
    bl_predictions.append(pred)
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{len(valid_rows)} done")

del bl_model, bl_processor
gc.collect()
print("  Baseline model unloaded from memory")

# ========== TEST 3: EASYOCR ==========
# Finally, I'll test the standard EasyOCR package as my third point of comparison
print("\n--- [3/3] Testing EASYOCR ---")
import easyocr
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

ez_predictions = []
for i, row in enumerate(valid_rows):
    results = reader.readtext(row['image'], detail=0)
    pred = ' '.join(results) if results else ''
    ez_predictions.append(pred)
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{len(valid_rows)} done")

del reader
gc.collect()
print("  EasyOCR unloaded from memory")

# ========== CALCULATE THE FINAL METRICS ==========
# Now I'll tally up all the scores and see who won!
print("\n" + "=" * 70)
print("COMPARISON RESULTS")
print("=" * 70)

ft_cers, bl_cers, ez_cers = [], [], []
ft_exact, bl_exact, ez_exact = 0, 0, 0

for i, row in enumerate(valid_rows):
    truth = row['text']
    
    ft_cer = calculate_cer(ft_predictions[i], truth)
    bl_cer = calculate_cer(bl_predictions[i], truth)
    ez_cer = calculate_cer(ez_predictions[i], truth)
    
    ft_cers.append(ft_cer)
    bl_cers.append(bl_cer)
    ez_cers.append(ez_cer)
    
    if ft_predictions[i].strip() == truth.strip(): ft_exact += 1
    if bl_predictions[i].strip() == truth.strip(): bl_exact += 1
    if ez_predictions[i].strip() == truth.strip(): ez_exact += 1

n = len(valid_rows)
ft_avg = sum(ft_cers) / n
bl_avg = sum(bl_cers) / n
ez_avg = sum(ez_cers) / n

print(f"\n{'Metric':<25} {'Fine-tuned':<18} {'Baseline':<18} {'EasyOCR':<18}")
print("-" * 70)
print(f"{'Avg CER':<25} {ft_avg*100:.2f}%{'':<12} {bl_avg*100:.2f}%{'':<12} {ez_avg*100:.2f}%")
print(f"{'Exact Matches':<25} {ft_exact}/{n} ({ft_exact/n*100:.0f}%){'':<6} {bl_exact}/{n} ({bl_exact/n*100:.0f}%){'':<6} {ez_exact}/{n} ({ez_exact/n*100:.0f}%)")

# Calculating how many times each model won outright (had the lowest error)
ft_wins = sum(1 for i in range(n) if ft_cers[i] <= bl_cers[i] and ft_cers[i] <= ez_cers[i])
bl_wins = sum(1 for i in range(n) if bl_cers[i] < ft_cers[i] and bl_cers[i] <= ez_cers[i])
ez_wins = sum(1 for i in range(n) if ez_cers[i] < ft_cers[i] and ez_cers[i] < bl_cers[i])

print(f"{'Wins (lowest CER)':<25} {ft_wins}/{n}{'':<12} {bl_wins}/{n}{'':<12} {ez_wins}/{n}")

# ========== MY SIDE-BY-SIDE VIEW ==========
# I like to print out a visual comparison of the predictions so I can see the exact typos
print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISONS")
print("=" * 70)

for i in range(min(10, n)):
    truth = valid_rows[i]['text']
    cers = {'FT': ft_cers[i], 'BL': bl_cers[i], 'EZ': ez_cers[i]}
    winner = min(cers, key=cers.get)
    
    print(f"\nSample {i+1} [WINNER: {winner}]")
    print(f"  Truth:      {truth}")
    print(f"  Fine-tuned: {ft_predictions[i]}  (CER={ft_cers[i]:.3f})")
    print(f"  Baseline:   {bl_predictions[i]}  (CER={bl_cers[i]:.3f})")
    print(f"  EasyOCR:    {ez_predictions[i]}  (CER={ez_cers[i]:.3f})")

# ========== MY FINAL VERDICT ==========
# Drumroll please...
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

best = min([('Fine-tuned TrOCR', ft_avg), ('Baseline TrOCR', bl_avg), ('EasyOCR', ez_avg)], key=lambda x: x[1])
print(f"\nAvg CER: Fine-tuned={ft_avg*100:.2f}% | Baseline={bl_avg*100:.2f}% | EasyOCR={ez_avg*100:.2f}%")
print(f"BEST MODEL: {best[0]} (CER: {best[1]*100:.2f}%)")
print(f"\nStep 16 Complete!")
