# AURA Project - Partner A Comprehensive Update

**Date**: December 25, 2024  
**From**: Partner B (The Analyst)  
**Project**: AURA - Affective Understanding for Robust Abuse Detection

---

## Executive Summary

Partner B has completed **all data preparation, preprocessing, and evaluation infrastructure**. All 4 datasets are downloaded, processed, and ready for model training. This document contains everything you need to start building the models.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Datasets Downloaded & Processed](#datasets-downloaded--processed)
3. [Critical Findings & Class Imbalance Analysis](#critical-findings--class-imbalance-analysis)
4. [Configuration Changes for Kaggle](#configuration-changes-for-kaggle)
5. [Problems Encountered & Fixes](#problems-encountered--fixes)
6. [Files You Need to Create](#files-you-need-to-create)
7. [Interface Contracts](#interface-contracts)
8. [How to Use the Data Pipeline](#how-to-use-the-data-pipeline)
9. [Timeline & Deliverables](#timeline--deliverables)

---

## 1. Project Structure

```
AURA/
├── venv/                          # Python virtual environment
├── config.yaml                    # All hyperparameters (UPDATED!)
├── requirements.txt               # Dependencies
│
├── data/
│   ├── raw/                       # Empty (using HuggingFace cache)
│   ├── processed/                 # ALL PROCESSED DATA (9 files, ~32MB)
│   └── splits/                    # Empty (splits in processed/)
│
├── scripts/                       # LOCAL RUN SCRIPTS (no GPU needed)
│   ├── run_eda.py                 # Exploratory Data Analysis
│   ├── run_preprocessing.py       # GoEmotions → Ekman mapping
│   ├── download_olid.py           # OLID dataset pipeline
│   └── download_jigsaw.py         # Jigsaw dataset pipeline
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py       # Text cleaning + Ekman mapping
│   │   ├── datasets.py            # PyTorch Dataset classes (needs torch)
│   │   └── loaders.py             # DataLoader + interleaving
│   ├── evaluation/
│   │   ├── metrics.py             # F1, confusion matrix, FPR
│   │   └── stress_test.py         # 3-Tier stress test
│   ├── models/                    # ⚠️ YOUR FILES GO HERE
│   │   └── __init__.py            # Stub with interface contract
│   └── training/                  # ⚠️ YOUR FILES GO HERE
│       └── __init__.py            # Stub with interface contract
│
├── outputs/
│   ├── figures/                   # 4 EDA figures generated
│   ├── models/                    # For saved model checkpoints
│   └── logs/                      # For training logs
│
├── notebooks/
│   └── 01_EDA.ipynb               # Original notebook (deprecated - use scripts/)
│
└── report/                        # For final report
```

---

## 2. Datasets Downloaded & Processed

All datasets are in `data/processed/` ready to use.

### 2.1 GoEmotions (Emotion Training)
| File | Samples | Purpose |
|------|---------|---------|
| `goemotions_ekman_train.json` | 43,410 | Emotion head training |
| `goemotions_ekman_val.json` | 5,426 | Validation |
| `goemotions_ekman_test.json` | 5,427 | Test |
| `class_weights.json` | 6 weights | For weighted loss |

**Labels**: 6 Ekman emotions (anger, disgust, fear, joy, sadness, surprise)

### 2.2 OLID (Toxicity Training - Stress Test Level 1)
| File | Samples | Purpose |
|------|---------|---------|
| `olid_train.json` | 11,280 | Toxicity head training |
| `olid_val.json` | 1,410 | Validation |
| `olid_test.json` | 1,410 | In-domain test (Level 1) |
| `olid_class_weights.json` | 2 weights | For weighted loss |

**Labels**: Binary (0=non-toxic, 1=toxic)  
**Source**: SemEval-2019 Task 6 (Twitter data)

### 2.3 Jigsaw (Stress Test Level 2)
| File | Samples | Purpose |
|------|---------|---------|
| `jigsaw_test.json` | 10,000 | Domain shift evaluation |

**Labels**: Binary (balanced 50/50)  
**Source**: Wikipedia comments (DIFFERENT domain from training!)

### 2.4 ToxiGen (Stress Test Level 3)
| Status | Details |
|--------|---------|
| Cached | Downloaded via HuggingFace during EDA |
| Samples | 8,960 annotated |
| Purpose | Implicit hate detection (hardest test!) |

**⚠️ ToxiGen is TEST ONLY - never use for training!**

---

## 3. Critical Findings & Class Imbalance Analysis

### 3.1 Ekman Emotion Distribution (SEVERE IMBALANCE!)

| Emotion | Train % | Class Weight | Risk Level |
|---------|---------|--------------|------------|
| **Joy** | 66.0% | 0.25 | ⚠️ Will dominate if not weighted |
| Anger | 12.9% | 1.30 | OK |
| Surprise | 11.4% | 1.46 | OK |
| Sadness | 6.8% | 2.47 | Low |
| **Fear** | 1.6% | **10.47** | 🔴 Severely underrepresented |
| **Disgust** | 1.5% | **11.34** | 🔴 CRITICAL for hate detection! |

**Why This Matters**:
- AURA's hypothesis is that **Anger + Disgust** are markers of toxic content
- But **Disgust has only 1.5% representation** in training data
- **You MUST use weighted loss** (weights in `class_weights.json`)
- **Focal Loss (γ=2)** will help focus on hard examples

### 3.2 OLID Toxicity Distribution

| Label | Train % | Class Weight |
|-------|---------|--------------|
| Non-toxic (NOT) | 67.1% | 0.75 |
| Toxic (OFF) | 32.9% | 1.52 |

**Moderate imbalance** - weighted loss recommended.

### 3.3 Vocabulary Analysis

| Comparison | Overlap | Implication |
|------------|---------|-------------|
| GoEmotions vs ToxiGen | 42.8% | Significant domain gap |

**Text Length Stats**:
- GoEmotions: ~12.8 words/sample (short Reddit comments)
- ToxiGen: ~18.1 words/sample (longer, machine-generated)
- BERT max_length=128 is sufficient for both

---

## 4. Configuration Changes for Kaggle

**File**: `config.yaml`

### 4.1 Training Config (CRITICAL CHANGES!)

```yaml
training:
  batch_size: 16                    # ⬅️ REDUCED from 32 (Kaggle P100 has 13GB VRAM)
  gradient_accumulation_steps: 2    # ⬅️ NEW (effective batch size = 32)
  mixed_precision: true             # ⬅️ NEW (FP16 for memory savings)
  epochs: 10
  learning_rate:
    encoder: 2e-5                   # BERT encoder
    heads: 1e-4                     # Task-specific heads
  warmup_ratio: 0.1                 # 10% of steps
  weight_decay: 0.01
  gradient_clip: 1.0
  early_stopping_patience: 3
```

### 4.2 Why These Changes?
- **P100 GPU (13GB VRAM)**: batch_size=32 causes OOM with BERT-base
- **Gradient Accumulation**: Simulates larger batch without memory cost
- **Mixed Precision**: ~40% memory reduction with FP16

---

## 5. Problems Encountered & Fixes

### 5.1 Fixed Issues ✅

| Issue | Problem | Fix |
|-------|---------|-----|
| **Batch Size OOM** | 32 too large for P100 | Reduced to 16 + gradient accumulation |
| **Neutral Label** | GoEmotions 'neutral' defaulted to 'joy' silently | Explicit handling in `map_goemotions_to_ekman()` |
| **Windows Encoding** | Unicode symbols crashed console | Added utf-8 reconfiguration |
| **Matplotlib Tk Error** | No display on Windows | Switched to 'Agg' backend |
| **Import Chain** | datasets.py imports torch | Created direct module imports in scripts |

### 5.2 Known Issues for Partner A to Handle ⚠️

| Issue | Description | Your Action |
|-------|-------------|-------------|
| **Disgust Underrepresentation** | Only 1.5% of emotion data | Use Focal Loss + Class Weights |
| **Domain Gap** | 42.8% vocabulary overlap | Stress test will reveal generalization |
| **Neutral → Joy Mapping** | May inflate Joy predictions | Consider alternative: 7th "neutral" class? |

### 5.3 Potential Future Improvements 💡

| Improvement | Description | Priority |
|-------------|-------------|----------|
| **Data Augmentation** | Back-translation for Disgust/Fear samples | Medium |
| **Curriculum Learning** | Train on easy samples first | Low |
| **Ensemble** | Multiple AURA models with different seeds | Low |

---

## 6. Files You Need to Create

### 6.1 `src/models/lstm.py` (Week 1)
```python
class BiLSTMClassifier(nn.Module):
    """
    Bi-LSTM baseline for toxicity classification.
    
    Architecture:
        GloVe Embeddings (300d) → BiLSTM (hidden=256) → Linear(512 → 2)
    
    Expected Output:
        logits: Tensor of shape (batch, 2)
    """
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_classes=2):
        ...
    
    def forward(self, input_ids, attention_mask=None):
        # Return logits (batch, 2)
        ...
```

### 6.2 `src/models/bert.py` (Week 2)
```python
class BERTClassifier(nn.Module):
    """
    Single-task BERT classifier.
    
    Architecture:
        BERT-base → [CLS] pooling → Linear(768 → 2)
    
    Expected Output:
        logits: Tensor of shape (batch, 2)
    """
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.1):
        ...
    
    def forward(self, input_ids, attention_mask=None):
        # Return logits (batch, 2)
        ...
```

### 6.3 `src/models/aura.py` (Week 3) - THE MAIN MODEL
```python
class AURAModel(nn.Module):
    """
    Multi-Task Learning model with emotion-aware toxicity detection.
    
    Architecture:
        Shared BERT Encoder
            ├── Toxicity Head: Linear(768 → 2)
            └── Emotion Head: Linear(768 → 6)
    
    Expected Output:
        tuple: (toxicity_logits, emotion_logits)
            - toxicity_logits: (batch, 2)
            - emotion_logits: (batch, 6)
    """
    def __init__(self, model_name='bert-base-uncased', dropout=0.1):
        ...
    
    def forward(self, input_ids, attention_mask=None):
        # MUST return tuple: (tox_logits, emo_logits)
        ...
```

### 6.4 `src/models/losses.py` (Week 3)
```python
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Use γ=2 for emotion head (per AURA_README.md)
    """
    def __init__(self, gamma=2.0, alpha=None):
        ...

class UncertaintyWeightedLoss(nn.Module):
    """
    Combines toxicity and emotion losses with learnable uncertainty weights.
    
    L = (1/2σ₁²) * L_tox + (1/2σ₂²) * L_emo + log(σ₁σ₂)
    
    σ₁, σ₂ are learnable parameters.
    """
    def __init__(self):
        self.log_sigma_tox = nn.Parameter(torch.zeros(1))
        self.log_sigma_emo = nn.Parameter(torch.zeros(1))
        ...
```

### 6.5 `src/training/trainer.py` (Week 1-2)
```python
class Trainer:
    """
    Training loop with:
    - Gradient accumulation (steps=2 from config)
    - Mixed precision (FP16)
    - Early stopping (patience=3)
    - LR warmup (10% of steps)
    - Gradient clipping (max_norm=1.0)
    - Model checkpointing
    """
    def __init__(self, model, train_loader, val_loader, config):
        ...
    
    def train_epoch(self):
        ...
    
    def validate(self):
        ...
    
    def fit(self, num_epochs):
        ...
```

---

## 7. Interface Contracts

### 7.1 Model Output Format

For compatibility with `src/evaluation/stress_test.py`:

```python
# BERT/LSTM models
outputs = model(input_ids, attention_mask)
# outputs.shape = (batch, 2)  OR  outputs.logits.shape = (batch, 2)

# AURA model
tox_logits, emo_logits = model(input_ids, attention_mask)
# tox_logits.shape = (batch, 2)
# emo_logits.shape = (batch, 6)
```

### 7.2 Expected Model Interface

```python
class YourModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Tensor of shape (batch, seq_len)
            attention_mask: Tensor of shape (batch, seq_len)
        
        Returns:
            BERT/LSTM: logits (batch, 2)
            AURA: tuple (tox_logits, emo_logits)
        """
        pass
```

---

## 8. How to Use the Data Pipeline

### 8.1 Loading Processed Data

```python
import json
from pathlib import Path

DATA_DIR = Path("data/processed")

# Load training data
with open(DATA_DIR / "olid_train.json") as f:
    olid_train = json.load(f)

with open(DATA_DIR / "goemotions_ekman_train.json") as f:
    goemotions_train = json.load(f)

# Each sample is a dict:
# olid_train[0] = {
#     'text': 'cleaned text',
#     'original_text': 'raw text',
#     'label': 0 or 1,
#     'label_name': 'non-toxic' or 'toxic'
# }
#
# goemotions_train[0] = {
#     'text': 'cleaned text',
#     'ekman_label': 'anger',
#     'ekman_id': 0  # (anger=0, disgust=1, fear=2, joy=3, sadness=4, surprise=5)
# }
```

### 8.2 Loading Class Weights

```python
# For emotion head
with open(DATA_DIR / "class_weights.json") as f:
    weights = json.load(f)
    
emotion_weights = torch.tensor(weights['weights'])  # [1.30, 11.34, 10.47, 0.25, 2.47, 1.46]

# For toxicity head
with open(DATA_DIR / "olid_class_weights.json") as f:
    weights = json.load(f)
    
tox_weights = torch.tensor(weights['weights'])  # [0.75, 1.52]
```

### 8.3 Using PyTorch Datasets (in src/data/datasets.py)

```python
from src.data.datasets import ToxicityDataset, EmotionDataset, AURAMultiTaskDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# For toxicity
tox_dataset = ToxicityDataset(olid_train, tokenizer, max_length=128)

# For emotion
emo_dataset = EmotionDataset(goemotions_train, tokenizer, max_length=128)

# For multi-task (AURA)
mtl_dataset = AURAMultiTaskDataset(olid_train, goemotions_train, tokenizer, max_length=128)
```

---

## 9. Timeline & Deliverables

| Week | Your Tasks (Partner A) | My Deliverables (Partner B) |
|------|------------------------|----------------------------|
| **Week 1** (Dec 23-29) | GloVe loader, BiLSTM, training loop | ✅ Data pipeline DONE |
| **Week 2** (Dec 30-Jan 5) | BERT tokenizer, Single-Task BERT | Evaluate BiLSTM |
| **Week 3** (Jan 6-12) | Focal Loss, AURA model | Evaluate BERT, Run stress tests |
| **Week 4** (Jan 13-19) | Report: Intro, Methods, Conclusions | Report: Experiments, Results |

---

## Quick Start Checklist

- [ ] Activate venv: `.\venv\Scripts\activate` (Windows)
- [ ] Install torch: `pip install torch torchvision torchaudio`
- [ ] Install transformers: `pip install transformers`
- [ ] Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Create `src/models/lstm.py`
- [ ] Create `src/training/trainer.py`
- [ ] Train BiLSTM on OLID

---

## Questions?

Contact me if you need:
- Clarification on any interface or data format
- Help with Kaggle notebook setup
- Explanation of the stress test protocol
- Changes to the data pipeline

**All data is ready. Let's build AURA! 🚀**
