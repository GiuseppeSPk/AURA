# AURA Results Comparison: Partner A vs Partner B

**Date**: January 9, 2026  
**Purpose**: Document discrepancies between claimed and reproduced results

---

## Executive Summary

We attempted to reproduce Partner A's reported training results. While our implementation follows the same theoretical framework (Kendall et al., 2018 Uncertainty-Weighted MTL), the final metrics differ significantly. This document requests Partner A's review to identify the cause.

---

## Results Comparison

### 3-Tier Stress Test Performance

| Level | Dataset | Partner A (Claimed) | Partner B (Reproduced) | Gap |
|-------|---------|---------------------|------------------------|-----|
| **L1** | OLID | **0.7917** | 0.7171 | -0.075 |
| **L2** | Jigsaw | **0.7744** | 0.6091 | -0.165 |
| **L3** | ToxiGen | **0.5168** | 0.4014 | -0.115 |

**ΔF1 Comparison**:
- Partner A: 0.7917 - 0.5168 = **0.275**
- Partner B: 0.7171 - 0.4014 = **0.316**

---

## Our Training Paradigm

### 1. Dataset Design
We used a **Paired MTL Dataset** where each sample has BOTH labels:

```python
# OLID samples → Add pseudo-emotion label (neutral=6)
{'text': "...", 'tox_label': 1, 'emo_label': 6}

# GoEmotions samples → Add pseudo-toxicity label (0=non-toxic)
{'text': "...", 'tox_label': 0, 'emo_label': 3}
```

**Rationale**: This allows computing BOTH losses on every batch, which is required for proper Kendall et al. uncertainty weighting.

### 2. Loss Function
```python
L = (1/2σ₁²) × L_tox + (1/2σ₂²) × L_emo + log(σ₁) + log(σ₂)
```
- Both losses computed simultaneously
- log(σ) parameters learned during training
- Final weights: tox=1.97, emo=1.74

### 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| GPU | Tesla P100-PCIE-16GB |
| Batch Size | 16 |
| Gradient Accumulation | 2 |
| Epochs | 5 |
| LR (BERT) | 2e-5 |
| LR (Heads) | 1e-4 |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Mixed Precision | FP16 |

### 4. Training Progression

| Epoch | Train Loss | Val Tox F1 | Val Emo F1 | Learned Weights |
|-------|------------|------------|------------|-----------------|
| 1 | 0.5219 | 0.6890 | 0.5460 | tox=1.10, emo=1.05 |
| 2 | 0.2082 | 0.7037 | 0.5631 | tox=1.30, emo=1.18 |
| 3 | -0.0094 | 0.7693 | 0.5488 | tox=1.53, emo=1.35 |
| 4 | -0.2310 | 0.7790 | 0.5732 | tox=1.76, emo=1.55 |
| 5 | -0.4174 | 0.7654 | 0.5902 | tox=1.97, emo=1.74 |

---

## Questions for Partner A

> **Please compare your training paradigm with ours and identify what may have caused the performance gap.**

### Specific Questions:

1. **Training Paradigm**: Did you use interleaved single-task batches or joint paired batches?

2. **Pseudo-Labeling**: How did you handle samples without both labels?
   - Did OLID samples have emotion labels?
   - Did GoEmotions samples have toxicity labels?

3. **Validation Set**: How was validation performed?
   - Separate toxicity and emotion validation sets?
   - Or a combined paired validation set?

4. **Data Preprocessing**: Any differences in text cleaning or tokenization?

5. **Threshold Tuning**: Was the 0.5 toxicity threshold tuned on validation data?

6. **Random Seed**: Was a fixed random seed used for reproducibility?

7. **Best Model Selection**: Which epoch was selected as the best model?

---

## Possible Causes of Discrepancy

| Factor | Our Approach | Potential Partner A Approach |
|--------|--------------|------------------------------|
| **Pseudo-labels** | OLID→neutral, GoEmo→non-toxic | Unknown |
| **Batch composition** | Both tasks per batch | Possibly interleaved? |
| **Validation** | Paired dataset | Possibly separate? |
| **Threshold** | Fixed 0.5 | Possibly tuned? |

---

## Next Steps

1. Partner A reviews this document
2. Partner A provides their training paradigm details
3. We identify the root cause of discrepancy
4. We agree on which results to report in the final paper

---

**Author**: Partner B  
**Request**: Please respond with your training paradigm details so we can resolve this discrepancy before the final submission.
