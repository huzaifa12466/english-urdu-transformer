
# 🌐 English to Urdu Translation Transformer

## 📌 Overview

This project implements a **Transformer model** for translating English sentences into Urdu using **PyTorch**. It follows the architecture from the paper _“Attention is All You Need”_ by Vaswani et al., incorporating:

- Multi-head attention  
- Positional encoding  
- Feed-forward layers  

This repo contains:

- 🧠 Transformer architecture in `transformer.py`  
- 🏋️‍♂️ Training logic in `train_transformer.py`  
- 📊 Dataset (not included): `english_to_urdu_dataset.xlsx`  

⚠️ **Note:**  
The current model uses only **one encoder and decoder layer** with a **limited dataset**, leading to **less accurate predictions**.

---

## 📁 Project Structure

```
project-root/
│
├── transformer.py           # Core transformer model
├── train_transformer.py     # Training pipeline
├── requirements.txt         # Python dependencies
└── english_to_urdu_dataset.xlsx  # English-Urdu dataset (external)
```

---

## 🧠 `transformer.py` – Model Architecture

### ✨ Components:

- **Utility Functions**
  - `get_device()`: Auto-selects CUDA or CPU
  - `scale_dot_product()`: Scaled attention with mask support

- **Positional Encoding**  
  Adds positional info using sin/cos functions.

- **Sentence Embedding**  
  Tokenizes, adds start/end tokens, applies positional encoding.

- **MultiheadAttention**  
  Parallel attention computation over multiple heads.

- **Encoder / Decoder Layers**  
  Includes self-attention, normalization, and feed-forward modules.

- **Transformer**  
  Final integration with encoder, decoder & linear output layer.

---

## 🏋️ `train_transformer.py` – Training Script

### 🧹 Preprocessing:

- Loads `english_to_urdu_dataset.xlsx` using `pandas`
- Filters invalid sentences (e.g., >200 chars or unknown tokens)
- Builds vocabularies with `START`, `END`, and `PAD` tokens

### 📦 Dataset:

- `TextDataset` for batching sentence pairs  
- `DataLoader` used for training

### 🧠 Masking:

- `create_masks()` generates:
  - Encoder padding mask
  - Look-ahead mask
  - Decoder cross-attention mask

### 🔁 Training:

- **Hyperparameters**: `d_model=512`, `num_layers=1`, `num_heads=8`
- **Loss**: `CrossEntropyLoss` with padding ignored
- **Optimizer**: `Adam`
- **Loop**: Trains for 10 epochs  
- **Evaluation**: Translates sample: _"should we go to the mall?"_

---

## ❗ Known Issues

### 🧼 Unclean Dataset:
- Inconsistent tokenization
- Missing or noisy translations
- Invalid characters outside vocab

### 🧠 Limited Model Capacity:
- Only 1 encoder and decoder layer  
- Standard is 6–8 layers for translation tasks

### ⏱️ Insufficient Training:
- 10 epochs not enough for convergence
- Learning rate may need tuning

### 📊 Small Dataset:
- Only 10,000 sentences used  
- More data = better generalization

---

## 🚀 Suggestions for Improvement

### 📈 Improve Dataset
- Clean punctuation, normalize Urdu diacritics
- Use larger corpora like OpenSubtitles

### 🧱 Expand Model
- Use 6–8 encoder/decoder layers
- Try `d_model=768+` or `num_heads=12`

### 🧪 Tune Training
- Train for 50+ epochs
- Add learning rate scheduler + early stopping

### 🔬 Hyperparameter Tuning
- Try `lr=0.0005`, `drop_prob=0.3`, larger batch size

### 🎯 Data Augmentation
- Back-translation or sentence paraphrasing

### 🔠 Tokenization
- Use **subword tokenization** (e.g., SentencePiece, BPE)

### 📏 Evaluation
- Add **BLEU score** computation
- Use a proper validation set

### 🧭 Inference
- Implement **beam search** instead of greedy decoding
- Fine-tune on domain-specific data

---

## 💻 Installation & Usage

### 📦 Prerequisites

```bash
pip install -r requirements.txt
```

**Requirements:**
- `torch>=2.0.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `openpyxl>=3.1.0`

### 📁 Dataset

- Place `english_to_urdu_dataset.xlsx` in the root directory  
- Or modify the path in `train_transformer.py`

### ▶️ Running the Model

```bash
python train_transformer.py
```

### 🖨️ Output

- Filtered sentence count
- Epoch, iteration, and loss per batch
- Urdu translations of sample sentences

---

## 🔮 Future Work

- 🧹 Clean and expand dataset
- 🧠 Use deeper model architectures
- 📏 Add BLEU and validation tracking
- 🌍 Deploy with a simple web interface

---

## 📌 Notes

⚠️ The current model is experimental and has limitations due to:

- A single encoder/decoder layer
- Small, noisy dataset
- Basic training loop

For better results, use:

- A clean, larger dataset
- Deeper model (6–8 layers)
- Subword tokenization
