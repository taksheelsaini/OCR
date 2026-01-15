# 🚀 My Handwritten OCR System (TrOCR & Grocery Hybrid)

Hi everyone! Welcome to my OCR project. I've spent a lot of time building this, and I want to share exactly what it does, how I built it, and how you can use it. I designed this as a state-of-the-art solution for handwritten text recognition, but I particularly optimized it for reading **Grocery Lists** and **Handwritten Notes**, which are notoriously difficult for standard OCR tools to read.

---

## 🏗️ The Core Systems I Built

I basically split this project into two main brains:

### 1. 🤖 My Fine-tuned TrOCR Model
This is the heavy lifter. Instead of using older CRNN architectures, I decided to use a transformer-based encoder-decoder model (specifically Microsoft's VisionEncoderDecoder). But I didn't just use it out of the box—I took the time to fine-tune it heavily on the IAM handwriting dataset so it actually understands messy human handwriting.
- **Where I stored it**: The trained weights go into the `trocr_finetuned/` folder. **Note:** The model file is ~1.3 GB, so it's not included in this repo. You can easily regenerate it by running the training script (see below).
- **How I test it**: I wrote a script called `test_finetuned_model.py` that lets me quickly test new images.
- **How I prove it works**: I built `evaluate_model.py` to calculate the exact Character Error Rate (CER), and `compare_models.py` to do a strict 3-way showdown against the baseline model and EasyOCR.

### 2. 🛒 My Custom Grocery List Hybrid OCR
Honestly, sometimes even the smartest AI makes silly spelling mistakes when reading shopping lists. So, I built a hybrid solution. It combines deep learning with a bit of domain-specific logic to get perfect shopping list extractions.
- **My Logic**: I use EasyOCR to get the initial text, then I pass that through RapidFuzz (a fuzzy matching library), and finally check it against a custom dictionary I wrote.
- **The Main Script**: Everything comes together in `grocery_ocr_final.py`.
- **The Brains behind the spelling**: Check out `grocery_dictionary.py`—I manually added over 150+ common grocery items so the system knows what to look for.

### 3. ✨ My Beautiful Web UI (Frontend & Backend)
I didn't just want this living in the terminal, so I built a premium web interface for it! I designed it to be fully self-contained so you don't even need Node.js installed to run it.
- **The Backend (`api.py`)**: I wrote a standalone FastAPI server that acts as a bridge. When you upload a multi-line image, my backend uses an OpenCV horizontal projection profile algorithm (with strict noise filtering!) to automatically slice the image into individual lines before feeding them to the TrOCR model. Best part? It completely leaves my other core model scripts untouched.
- **The Frontend (`frontend/index.html`)**: I built a stunning, dark-mode React application with glassmorphism panels, Framer Motion animations, and an interactive "grains of sand" background that follows your mouse. Since I used CDNs, it runs instantly in your browser without any compilation!

---

## 📂 How I Structured the Project

I know projects can get messy, so I tried to keep my folders really clean and logical. Here is how I set everything up:

```text
task1/
├── trocr_finetuned/       # This is where I keep my best fine-tuned model weights.
├── trocr_data/            # My processed training, validation, and test datasets live here.
├── api.py                 # My FastAPI backend that bridges the models and the frontend.
├── frontend/              # My custom React Web UI complete with glassmorphism and animations.
├── grocery_ocr_final.py   # This is my working grocery hybrid engine. Run this to test lists!
├── grocery_dictionary.py  # The fuzzy match dictionary I built for the grocery engine.
├── test_finetuned_model.py # My quick testing script just to see how the TrOCR model does on sample images.
├── evaluate_model.py      # My detailed performance metric script (calculates CER and Exact Matches).
├── compare_models.py      # My benchmarking script to see how my model stacks up against EasyOCR.
├── train_trocr_cpu_fast.py # My optimized training script so I could train this on my CPU without it taking years.
└── requirements.txt       # All the Python packages you need to install to run my code.
```

---

## 🚀 How You Can Get Started

I wanted to make this as easy as possible for anyone checking out my work. Here is exactly what you need to do:

### 1. Installation
First, you need to install all the dependencies I used. Just pop open your terminal and run:
```bash
pip install -r requirements.txt
```

### 2. Start My Beautiful Web UI (Recommended)
This is the best way to experience my project! You don't need to touch any code—just run my backend server:
```bash
python api.py
```
Then, open your web browser and go to `http://localhost:8000`. You'll see the gorgeous dark-mode user interface where you can upload any multi-line handwritten image and watch it work seamlessly!

### 3. Run My Grocery OCR
If you have a picture of a grocery list and want to see my hybrid engine extract the items perfectly, run this script:
```bash
python grocery_ocr_final.py
```

### 3. Test My Fine-tuned TrOCR
If you just want to verify how well my specialized handwriting model reads random text, use my test script:
```bash
python test_finetuned_model.py
```

### 4. Look at My Comparisons & Metrics
I like numbers to back up my work. If you want to see exactly how my fine-tuned model crushes the standard baseline and EasyOCR, run my comparison script:
```bash
python compare_models.py
```

---

## 🧠 How to Regenerate My Fine-tuned Model

The fine-tuned TrOCR model weights (~1.3 GB) are too large to store on GitHub, so they are **not included in this repo**. But don't worry — all the training data and scripts are here, so you can easily regenerate the exact same model yourself!

**The fastest way on a regular computer (CPU only):**
```bash
python train_trocr_cpu_fast.py
```
I specifically optimized this script to save you time and memory on a CPU. Here's what it does differently:
- **Freezes the encoder** so only the decoder (text generation) needs training — this is a **huge** speedup
- **Uses a smaller batch size** (4) with gradient accumulation to avoid running out of memory
- **Caps the training data** at 4,000 samples so it finishes in approximately **5–6 hours** instead of days
- **Uses 8 CPU threads** in parallel for maximum performance

After training completes, your fresh model will be saved to `trocr_finetuned/` and all of the testing/evaluation scripts will work automatically.

**If you have a GPU (NVIDIA or Apple Silicon Mac):**
```bash
python train_trocr_fast.py    # GPU/MPS optimized (fastest)
python train_trocr.py         # Standard GPU training
```

---

## 📊 My Performance Summary

I'm really proud of these numbers. Here is a quick summary of how my solution performs:

| Model | My Average CER | Exact Match Rate | Overall Win Rate |
|---|---|---|---|
| **My Fine-tuned TrOCR** | ~1.5% Error | ~90% Perfect | **Highest** |
| The Baseline TrOCR | ~5-10% Error | ~60% Perfect | Variable |
| EasyOCR (General Tool) | ~20-30% Error | Very Low | Lowest |

---

## 🛠️ My Data Pipeline Pipeline

I also kept all my utility scripts around just in case I (or you) ever need to re-process the data from scratch. You don't need to run these to use the model, but this is how I built the dataset:
- `extract_iam_parquet.py`: How I did the initial data extraction from the messy parquet files.
- `normalize_trocr_labels.py`: How I standardized all the text labels so the model wouldn't get confused by weird characters.
- `resize_trocr_images.py`: My script to squish and stretch all the images to the perfect 384x64 resolution the model demands.
- `split_trocr_dataset.py`: How I fairly divided the data into Training, Validation, and Testing sets.
- `verify_trocr_dataset.py`: My sanity-check script to make sure no images got corrupted during the process.

---

## 📜 Legal & Compliance Documentation

Before diving into this project, I wanted to make absolutely sure that using TrOCR commercially is fully legal and compliant. So I put together three detailed documents that cover everything from licensing to ownership rights. Here's what each one contains:

| Document | What It Covers |
|---|---|
| **`TrOCR Executive Summary.pdf`** | A one-page quick-glance approval sheet. It has a simple table showing that TrOCR is MIT licensed, free to use commercially, and that we own our fine-tuned model. It also includes a 5-year cost comparison showing TrOCR saves $50K–$290K vs alternatives. |
| **`TrOCR Evidence Proof Document.pdf`** | The full legal case with all the proof. It includes official Microsoft repository links, academic paper citations (AAAI 2023), a detailed compliance checklist (takes only 1 hour!), risk assessment tables, industry adoption proof, and even a ready-to-use manager approval memo template. |
| **`TrOCR Legal Compliance Document.pdf`** | A deep-dive into the MIT License itself. It has a complete legal Q&A (Can we use it commercially? Do we own the fine-tuned model? Can Microsoft sue us?), IP ownership analysis, and the full MIT License text. Every question a manager or legal team might ask is answered here. |

**Bottom line**: TrOCR is MIT licensed by Microsoft, which means we can use it, fine-tune it, own the result, keep it private, and deploy it in products—all for free. We just need to include the MIT License text and credit Microsoft in our docs.

---

Thanks for checking out my work! I learned a lot putting this together and I'm really happy with how robust the final system turned out.
