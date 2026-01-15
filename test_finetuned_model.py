"""
This is my quick testing script that I wrote to try out my fine-tuned TrOCR model on some sample images.
It loads my trained weights and simply runs inference so I can see the results with my own eyes!
"""
import os
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pandas as pd

# Here I'm pointing the script to the directory where I saved my trained model weights
MODEL_PATH = 'trocr_finetuned'
print(f"Loading my fine-tuned model from {MODEL_PATH}...")

processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"✓ Model loaded on {device}")

def predict_text(image_path):
    """
    My custom function to predict text from a single image.
    I pass in the path to the image, and it returns the recognized text string.
    """
    image = Image.open(image_path).convert('RGB')
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    
    with torch.no_grad():
        generated = model.generate(pixel_values)
    
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return text

# Now I'm going to test my model on a few random samples from my test dataset
print("\n" + "="*60)
print("TESTING MY MODEL ON SAMPLE IMAGES")
print("="*60)

test_csv = 'trocr_data/final_test_labels.csv'
if os.path.exists(test_csv):
    # I load my test metadata using pandas
    test_df = pd.read_csv(test_csv)
    
    # I just want to loop through the first 5 samples so I'm not waiting forever
    for i in range(min(5, len(test_df))):
        row = test_df.iloc[i]
        image_path = row['image']
        ground_truth = row['text']
        
        if os.path.exists(image_path):
            prediction = predict_text(image_path)
            
            print(f"\nSample {i+1}:")
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Prediction:   {prediction}")
            
            # Here I do a really simple string comparison just to see if it got it 100% right.
            if prediction.strip().lower() == ground_truth.strip().lower():
                print(f"  Result: ✓ MATCH")
            else:
                print(f"  Result: ✗ Different")
else:
    # Fallback option just in case my CSV is missing
    print(f"Test CSV not found: {test_csv}")
    print("Testing on images in my test_images folder...")
    
    if os.path.exists('test_images'):
        for img_file in os.listdir('test_images')[:5]:
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join('test_images', img_file)
                prediction = predict_text(img_path)
                print(f"\n{img_file}: {prediction}")

print("\n" + "="*60)
print("✅ My testing script is complete!")
print("="*60)
print(f"\nMy fine-tuned model is waiting for you at: {MODEL_PATH}")
print("You can import the predict_text() function into your own scripts if you want me to read something else!")
