import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
import evaluate
import random

# Setting up my device
device = torch.device("cpu")
print("⚠️ I'm using my CPU here - I optimized this script specifically so it wouldn't take a million years to train!")

# I found out that PyTorch doesn't automatically use all your CPU cores, so I force it to use 8 threads here
torch.set_num_threads(8)

# ========== CONFIGURATION ==========
MODEL_NAME = 'microsoft/trocr-base-handwritten'
OUTPUT_DIR = 'trocr_finetuned'
TRAIN_CSV = 'trocr_data/final_train_labels.csv'
VAL_CSV = 'trocr_data/final_val_labels.csv'
TEST_CSV = 'trocr_data/final_test_labels.csv'

# ========== MY CPU-OPTIMIZED HYPERPARAMETERS ==========
# Since I'm on a CPU, I had to use a really small batch size so it doesn't instantly crash from running out of memory
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 4 * 4 = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3  # Increased from 2 for better accuracy
MAX_LENGTH = 32  # Reduced from 64 (shorter sequences = faster)

# I'm using a fairly large subset of the data because I want it to actually learn, but I still cap it so training finishes in ~5 hours
MAX_TRAIN_SAMPLES = 4000
MAX_VAL_SAMPLES = 400  # Increased from 200

# I dialed back how frequently it evaluates and saves because doing that on a CPU takes forever
SAVE_STEPS = 300
EVAL_STEPS = 300
LOGGING_STEPS = 25

# ========== LOAD MY MODEL & PROCESSOR ==========
print("Loading my base TrOCR model and processor...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Keep on CPU
model = model.to(device)

# Set special tokens
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id

# I have to manually set the generation config because the newer versions of transformers complain if I don't
model.generation_config.max_length = MAX_LENGTH
model.generation_config.early_stopping = True
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty = 2.0
model.generation_config.num_beams = 2  # Reduced from 4 for speed

# ========== FREEZE ENCODER (MY SECRET TO FAST TRAINING!) ==========
print("Freezing the encoder layers so I only have to train the text generation part...")
for param in model.encoder.parameters():
    param.requires_grad = False

# Counting how many parameters I actually have to calculate gradients for
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
print("Model loaded successfully!")

# ========== MY OPTIMIZED DATASET CLASS ==========
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
        
        # Load and process image (resize smaller for speed)
        image = Image.open(img_path).convert('RGB')
        
        # Resize to smaller size for faster processing
        image = image.resize((384, 384), Image.Resampling.LANCZOS)
        
        pixel_values = self.processor(image, return_tensors='pt').pixel_values.squeeze()
        
        # Process text (labels)
        encoding = self.processor.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        labels = encoding.input_ids.squeeze()
        
        # Create decoder_input_ids
        decoder_input_ids = labels.clone()
        decoder_input_ids[1:] = labels[:-1].clone()
        decoder_input_ids[0] = self.processor.tokenizer.cls_token_id
        
        # Replace padding with -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }

# ========== LOAD DATASETS WITH SUBSET ==========
print("Loading datasets...")
full_train_dataset = OCRDataset(TRAIN_CSV, processor)
full_val_dataset = OCRDataset(VAL_CSV, processor)

# Use random subset for faster training
random.seed(42)
train_indices = random.sample(range(len(full_train_dataset)), min(MAX_TRAIN_SAMPLES, len(full_train_dataset)))
val_indices = random.sample(range(len(full_val_dataset)), min(MAX_VAL_SAMPLES, len(full_val_dataset)))

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)

print(f"Train: {len(train_dataset)} samples (subset), Val: {len(val_dataset)} samples (subset)")

# ========== MY EVALUATION METRICS ==========
cer_metric = evaluate.load('cer')

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {'cer': cer}

# ========== MY CPU-OPTIMIZED TRAINING ARGUMENTS ==========
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Smaller batch for CPU
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    
    # Fewer epochs
    num_train_epochs=NUM_EPOCHS,
    
    # Learning rate
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    lr_scheduler_type='linear',
    
    # Less frequent eval/save
    eval_strategy='steps',
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    logging_steps=LOGGING_STEPS,
    
    # Best model
    load_best_model_at_end=True,
    metric_for_best_model='cer',
    greater_is_better=False,
    
    # Generation
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    
    # CPU optimizations
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    fp16=False,  # No mixed precision on CPU
    
    # Optimizer
    optim='adamw_torch',
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # No external logging
    report_to='none',
    
    # Label smoothing
    label_smoothing_factor=0.1,
)

# ========== TRAINER ==========
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor.tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# ========== TRAINING INFO ==========
total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NUM_EPOCHS
estimated_time_hours = (total_steps * 15) / 3600  # ~15 sec per step on CPU

print("\n" + "="*60)
print("🚀 STARTING MY CPU-OPTIMIZED FINE-TUNING RUN")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Device: {device}")
print(f"Train samples: {len(train_dataset)} (subset of {len(full_train_dataset)})")
print(f"Val samples: {len(val_dataset)} (subset of {len(full_val_dataset)})")
print(f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} effective")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Total steps: ~{total_steps}")
print(f"Estimated time: ~{estimated_time_hours:.1f} hours")
print(f"Encoder frozen: Yes (faster training)")
print("="*60 + "\n")

# ========== TRAIN ==========
trainer.train()

# ========== SAVE FINAL MODEL ==========
print("\n✅ Saving fine-tuned model...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Model saved to: {OUTPUT_DIR}")

# ========== FINAL EVALUATION ==========
print("\n📊 Evaluating on test set...")
test_dataset = OCRDataset(TEST_CSV, processor)
# Use subset for faster eval
test_indices = random.sample(range(len(test_dataset)), min(200, len(test_dataset)))
test_subset = Subset(test_dataset, test_indices)
test_results = trainer.evaluate(test_subset)
print(f"Test CER: {test_results['eval_cer']:.4f}")
print(f"(Lower is better - 0.0 = perfect, 0.1 = 10% character errors)")

print("\n" + "="*60)
print("🎉 MY FINE-TUNING IS COMPLETE! Time for a coffee break.")
print("="*60)
