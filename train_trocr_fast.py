"""
My Lightning-Fast TrOCR Fine-tuning Script!
I built this version specifically to take advantage of GPU/MPS acceleration so I don't have to wait days for training.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
import evaluate

# ========== MY DEVICE SETUP ==========
# This is my favorite part: it automatically detects if I'm on a Mac with Apple Silicon (MPS)
# or a PC with an NVIDIA card (CUDA) and accelerates the training!
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using MPS (Apple Silicon GPU) - FAST MODE")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA GPU - FAST MODE")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU - Training will be slower")

# ========== MY CONFIGURATION SETTINGS ==========
MODEL_NAME = 'microsoft/trocr-base-handwritten'
OUTPUT_DIR = 'trocr_finetuned'
TRAIN_CSV = 'trocr_data/final_train_labels.csv'
VAL_CSV = 'trocr_data/final_val_labels.csv'
TEST_CSV = 'trocr_data/final_test_labels.csv'

# ========== MY HIGH-SPEED HYPERPARAMETERS ==========
# Since I'm using a GPU here, I can crank up the batch size and use gradient accumulation
# This gives me faster training AND better accuracy!
BATCH_SIZE = 16  # Increased from 8
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 16 * 2 = 32
LEARNING_RATE = 4e-5  # Slightly lower for stability with larger batch
NUM_EPOCHS = 3
MAX_LENGTH = 64  # Reduced from 128 (most texts are shorter)
WARMUP_RATIO = 0.1  # Use ratio instead of fixed steps
SAVE_STEPS = 500
EVAL_STEPS = 250  # More frequent eval to catch best model
LOGGING_STEPS = 50

# ========== LOAD MY MODEL & PROCESSOR ==========
print("Loading my base TrOCR model and processor...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Move model to device
model = model.to(device)

# Set special tokens
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id

# Set generation config (required for newer transformers versions)
model.generation_config.max_length = MAX_LENGTH
model.generation_config.early_stopping = True
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty = 2.0
model.generation_config.num_beams = 4

# ========== FREEZE ENCODER (MY TRANSFER LEARNING TRICK) ==========
# The vision encoder is already super smart from reading millions of images.
# I freeze it here so my computer only has to spend time training the text decoder!
print("Freezing encoder layers for faster training...")
for param in model.encoder.parameters():
    param.requires_grad = False

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
print("Model loaded successfully!")

# ========== MY HIGH-SPEED CACHING DATASET ==========
# I wrote this custom Dataset class with a built-in cache so it doesn't have to 
# constantly read the same images from my hard drive over and over again.
class OCRDataset(Dataset):
    def __init__(self, csv_path, processor, max_length=MAX_LENGTH):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.max_length = max_length
        self.cache = {}  # Cache processed samples
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
            
        row = self.df.iloc[idx]
        img_path = row['image']
        text = str(row['text'])
        
        # Load and process image
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.processor(image, return_tensors='pt').pixel_values.squeeze()
        
        # Process text (labels)
        encoding = self.processor.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input_ids for labels
        labels = encoding.input_ids.squeeze()
        
        # Create decoder_input_ids (shifted right with decoder_start_token)
        decoder_input_ids = labels.clone()
        decoder_input_ids[1:] = labels[:-1].clone()
        decoder_input_ids[0] = self.processor.tokenizer.cls_token_id  # Start token
        
        # Replace padding token id with -100 in labels so it's ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        result = {
            'pixel_values': pixel_values,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }
        
        # Cache if memory allows (first 1000 samples)
        if idx < 1000:
            self.cache[idx] = result
            
        return result

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
    
    # Compute CER (Character Error Rate)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {'cer': cer}

# ========== MY HIGH-SPEED TRAINING ARGUMENTS ==========
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Batch settings (larger effective batch = faster convergence)
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    
    # Training duration
    num_train_epochs=NUM_EPOCHS,
    
    # Learning rate with warmup
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type='cosine',  # Cosine decay = better final accuracy
    
    # Evaluation & saving
    eval_strategy='steps',
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,  # Keep only 2 checkpoints to save space
    logging_steps=LOGGING_STEPS,
    
    # Best model tracking
    load_best_model_at_end=True,
    metric_for_best_model='cer',
    greater_is_better=False,
    
    # Generation settings
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    
    # Performance optimizations
    dataloader_num_workers=0,  # macOS compatibility
    dataloader_pin_memory=False,  # MPS compatibility
    
    # Gradient optimizations
    optim='adamw_torch',  # Fast optimizer
    weight_decay=0.01,  # Regularization for better accuracy
    max_grad_norm=1.0,  # Gradient clipping for stability
    
    # Disable external logging
    report_to='none',
    
    # Speed optimizations
    torch_compile=False,  # Set True if PyTorch 2.0+ and not MPS
    
    # Label smoothing for better generalization
    label_smoothing_factor=0.1,
)

# ========== CALLBACKS ==========
# Early stopping prevents overfitting and saves time
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # Stop if no improvement for 3 evals
    early_stopping_threshold=0.01  # Minimum improvement required
)

# ========== TRAINER ==========
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor.tokenizer,  # Updated from tokenizer
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

# ========== TRAINING INFO ==========
total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NUM_EPOCHS
print("\n" + "="*60)
print("🚀 STARTING MY HIGH-SPEED FINE-TUNING RUN")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Device: {device}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} effective")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Total steps: ~{total_steps}")
print(f"Learning rate: {LEARNING_RATE} (cosine decay)")
print(f"Encoder frozen: Yes (faster training)")
print(f"Early stopping: After 3 evals without improvement")
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
test_results = trainer.evaluate(test_dataset)
print(f"Test CER: {test_results['eval_cer']:.4f}")
print(f"(Lower is better - 0.0 = perfect, 0.1 = 10% character errors)")

print("\n" + "="*60)
print("🎉 MY HIGH-SPEED FINE-TUNING IS COMPLETE!")
print("="*60)
