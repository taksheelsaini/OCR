"""
My Primary TrOCR Fine-tuning Script!
I wrote this to train my model on the IAM handwritten dataset.
"""

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from datasets import Dataset as HFDataset
import evaluate

# ========== MY CONFIGURATION SETTINGS ==========
# I'm starting with the base model from Microsoft because it's already pretty smart
MODEL_NAME = 'microsoft/trocr-base-handwritten'
OUTPUT_DIR = 'trocr_finetuned'
TRAIN_CSV = 'trocr_data/final_train_labels.csv'
VAL_CSV = 'trocr_data/final_val_labels.csv'
TEST_CSV = 'trocr_data/final_test_labels.csv'

# My Training Hyperparameters
# I tweaked these a lot to get the best results without crashing my GPU
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 128
WARMUP_STEPS = 500
SAVE_STEPS = 1000
EVAL_STEPS = 500
LOGGING_STEPS = 100

# ========== LOAD MY MODEL & PROCESSOR ==========
print("Loading my base TrOCR model and processor...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# I have to set up these special tokens so the decoder knows how to structure sentences
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = MAX_LENGTH
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

print("Model loaded successfully!")

# ========== MY CUSTOM DATASET CLASS ==========
class OCRDataset(Dataset):
    def __init__(self, csv_path, processor, max_length=MAX_LENGTH):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image']
        text = str(row['text'])
        
        # I load the image and force it to RGB just in case there are any weird grayscale formats
        image = Image.open(img_path).convert('RGB')
        
        # Now I run the image through my processor to get the mathematical pixel values
        pixel_values = self.processor(image, return_tensors='pt').pixel_values.squeeze()
        
        # Process text (labels)
        labels = self.processor.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids.squeeze()
        
        # This is super important: I replace padding tokens with -100 so the model doesn't get confused 
        # and try to learn how to predict "blank" space
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }

# ========== LOAD MY DATASETS ==========
print("Loading my datasets into memory...")
train_dataset = OCRDataset(TRAIN_CSV, processor)
val_dataset = OCRDataset(VAL_CSV, processor)
print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

# ========== MY EVALUATION METRICS ==========
cer_metric = evaluate.load('cer')

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # Replace -100 with pad token id
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    # Compute CER
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {'cer': cer}

# ========== TRAINING ARGUMENTS ==========
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    eval_strategy='steps',
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    fp16=torch.cuda.is_available(),  # Use fp16 if GPU available
    dataloader_num_workers=0,  # Set to 0 for macOS compatibility
    dataloader_pin_memory=False,  # Disable pin_memory for MPS
    load_best_model_at_end=True,
    metric_for_best_model='cer',
    greater_is_better=False,
    report_to='none',  # Disable wandb/tensorboard
)

# ========== RUN THE TRAINER ==========
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# ========== START TRAINING ==========
print("\n" + "="*50)
print("STARTING MY FINE-TUNING RUN")
print("="*50)
print(f"Model: {MODEL_NAME}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*50 + "\n")

trainer.train()

# ========== SAVE FINAL MODEL ==========
print("\nSaving fine-tuned model...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Model saved to: {OUTPUT_DIR}")

# ========== EVALUATE ON TEST SET ==========
print("\nEvaluating on test set...")
test_dataset = OCRDataset(TEST_CSV, processor)
test_results = trainer.evaluate(test_dataset)
print(f"Test CER: {test_results['eval_cer']:.4f}")

print("\n" + "="*50)
print("MY FINE-TUNING IS COMPLETE!")
print("="*50)
