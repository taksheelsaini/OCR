"""
Phase 5: This is my comprehensive evaluation script for the fine-tuned TrOCR model.
I wrote this to test my model on the full test set, calculate the Character Error Rate (CER), 
and show some sample predictions so I can see exactly where it fails or succeeds.
"""

import os
import sys
import pandas as pd
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import random

# I had to fix the Windows encoding here so it prints symbols nicely in the terminal
sys.stdout.reconfigure(encoding='utf-8')

# ========== MY CONFIGURATION PARAMETERS ==========
# This is where I stored my trained model
MODEL_DIR = 'trocr_finetuned'
TEST_CSV = 'trocr_data/final_test_labels.csv'
NUM_SAMPLES = 20  # Number of sample predictions to show

# ========== LOAD MY MODEL ==========
print("Loading my fine-tuned model...")
device = torch.device('cpu')
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# ========== LOAD MY TEST DATA ==========
test_df = pd.read_csv(TEST_CSV)
print(f"Test set: {len(test_df)} samples")

# ========== MY PREDICTION ALGORITHM ==========
def predict_text(image_path):
    """Predict text from an image using my fine-tuned TrOCR model"""
    image = Image.open(image_path).convert('RGB')
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64, num_beams=4)
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

# ========== CALCULATE ERROR RATE (CER) ==========
def calculate_cer(pred, truth):
    """Calculate the Character Error Rate between my prediction and the ground truth"""
    import editdistance
    if len(truth) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, truth) / len(truth)

def calculate_cer_simple(pred, truth):
    """I wrote this simple CER calculating function so I don't have to rely on the editdistance library"""
    if len(truth) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    
    # I'm using dynamic programming here to find the minimum edit distance
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

# ========== EVALUATE ON MY FULL TEST SET ==========
print("\n" + "=" * 60)
print("EVALUATING MY MODEL ON THE FULL TEST SET")
print("=" * 60)

total_cer = 0
exact_matches = 0
total = 0
errors = 0
results = []

for idx, row in test_df.iterrows():
    img_path = row['image']
    truth = str(row['text'])
    
    if not os.path.exists(img_path):
        errors += 1
        continue
    
    try:
        pred = predict_text(img_path)
        cer = calculate_cer_simple(pred, truth)
        total_cer += cer
        total += 1
        
        if pred.strip() == truth.strip():
            exact_matches += 1
        
        results.append({
            'image': os.path.basename(img_path),
            'truth': truth,
            'prediction': pred,
            'cer': cer,
            'exact_match': pred.strip() == truth.strip()
        })
        
        # Progress
        if total % 50 == 0:
            avg_cer = total_cer / total
            print(f"  Processed {total}/{len(test_df)} | Avg CER: {avg_cer:.4f} | Exact matches: {exact_matches}/{total}")
            
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"  Error on {img_path}: {e}")

# ========== MY RESULTS SUMMARY ==========
# Here I calculate the final averages after processing the whole set=
avg_cer = total_cer / total if total > 0 else 0
exact_match_rate = exact_matches / total * 100 if total > 0 else 0

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Total samples evaluated: {total}")
print(f"Errors/skipped: {errors}")
print(f"")
print(f"Character Error Rate (CER): {avg_cer:.4f} ({avg_cer*100:.2f}%)")
print(f"Exact Match Rate: {exact_matches}/{total} ({exact_match_rate:.1f}%)")
print("=" * 60)

# Here is a quick CER breakdown so I know if the errors are minor typos or completely garbled
low_cer = sum(1 for r in results if r['cer'] < 0.05)
med_cer = sum(1 for r in results if 0.05 <= r['cer'] < 0.2)
high_cer = sum(1 for r in results if r['cer'] >= 0.2)

print(f"CER < 5% (Good):     {low_cer}/{total} ({low_cer/total*100:.1f}%)")
print(f"CER 5-20% (OK):      {med_cer}/{total} ({med_cer/total*100:.1f}%)")
print(f"CER > 20% (Bad):     {high_cer}/{total} ({high_cer/total*100:.1f}%)")

# ========== SHOW SOME SAMPLE PREDICTIONS ==========
# I like to print out a few random samples just so I can manually verify the numbers
print("\n" + "=" * 60)
print(f"SAMPLE PREDICTIONS ({NUM_SAMPLES} random)")
print("=" * 60)

# Let's grab some random samples
random.seed(42)
sample_indices = random.sample(range(len(results)), min(NUM_SAMPLES, len(results)))

for i, idx in enumerate(sample_indices):
    r = results[idx]
    match_symbol = "MATCH" if r['exact_match'] else f"CER={r['cer']:.3f}"
    print(f"\nSample {i+1}:")
    print(f"  Truth:      {r['truth']}")
    print(f"  Prediction: {r['prediction']}")
    print(f"  Result:     {match_symbol}")

# ========== MY WORST PREDICTIONS ==========
# This is the most helpful part for me: looking at where the model completely failed so I can improve it later.
print("\n" + "=" * 60)
print("TOP 5 WORST PREDICTIONS (highest CER)")
print("=" * 60)

worst = sorted(results, key=lambda x: x['cer'], reverse=True)[:5]
for i, r in enumerate(worst):
    print(f"\n#{i+1} (CER={r['cer']:.3f}):")
    print(f"  Truth:      {r['truth']}")
    print(f"  My Prediction: {r['prediction']}")

# ========== FINAL PERFORMANCE RATING ==========
# I wrote this little grading system to quickly tell me if I should be happy with the results or not.
print("\n" + "=" * 60)
print("PERFORMANCE RATING")
print("=" * 60)

if avg_cer < 0.03:
    rating = "EXCELLENT"
elif avg_cer < 0.05:
    rating = "VERY GOOD"
elif avg_cer < 0.10:
    rating = "GOOD"
elif avg_cer < 0.20:
    rating = "FAIR"
else:
    rating = "NEEDS IMPROVEMENT"

print(f"\nOverall Rating: {rating}")
print(f"CER: {avg_cer:.4f}")
print(f"Exact Match: {exact_match_rate:.1f}%")
print(f"\nModel location: {os.path.abspath(MODEL_DIR)}")
print(f"\nPhase 5 Evaluation Complete!")
