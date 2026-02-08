# AURA Results Comparison: Complete Analysis

**Date**: January 9, 2026  
**Purpose**: Comprehensive comparison of all training runs and Partner A's claims

---

## Executive Summary

We tested multiple epochs (3, 4, 5) with threshold tuning. **Epoch 5 with threshold 0.45 produced the best cross-domain generalization** (lowest ΔF1), though all results differ from Partner A's claims.

---

## 1. Complete Results Comparison

### Our Results by Epoch

| Epoch | Threshold | L1 (OLID) | L2 (Jigsaw) | L3 (ToxiGen) | ΔF1 |
|-------|-----------|-----------|-------------|--------------|-----|
| **3** | 0.25 | 0.7235 | 0.6145 | 0.4025 | 0.3210 |
| **4** | 0.25 | 0.7183 | 0.6396 | 0.4026 | 0.3157 |
| **5** | 0.45 | 0.7096 | **0.7346** | **0.4306** | **0.2791** |

### 🏆 Best Result: Epoch 5

```
Level 1 (OLID):   F1 = 0.7096
Level 2 (Jigsaw): F1 = 0.7346  ← Excellent cross-domain!
Level 3 (ToxiGen): F1 = 0.4306  
Delta F1: 0.2791  ← Best generalization
```

---

### Comparison with Partner A's Claims

| Level | Partner A | Our Best (Ep5) | Gap | Status |
|-------|-----------|----------------|-----|--------|
| **L1** | 0.7917 | 0.7096 | -0.082 | 🟡 -8.2 pp |
| **L2** | 0.7744 | 0.7346 | -0.040 | 🟢 -4.0 pp |
| **L3** | 0.5168 | 0.4306 | -0.086 | 🟡 -8.6 pp |
| **ΔF1** | 0.2749 | 0.2791 | +0.004 | 🟢 Nearly identical! |

**Key Finding**: Our **ΔF1 (0.2791) nearly matches Partner A's (0.2749)**!  
The cross-domain generalization is comparable, but absolute performance differs.

---

## 2. Detailed Classification Reports

### Epoch 5 (Best Overall)

| Dataset | Threshold | Non-Toxic Recall | Toxic Recall | Macro F1 |
|---------|-----------|------------------|--------------|----------|
| **OLID** | 0.45 | 74% | 70% | 0.71 |
| **Jigsaw** | 0.45 | 93% | 56% | 0.73 |
| **ToxiGen** | 0.45 | 99% | 6% | 0.43 |

### Toxic Recall Comparison

```
            Epoch 3 (0.25)  Epoch 4 (0.25)  Epoch 5 (0.45)
OLID:       59%             56%             70%  ✓
Jigsaw:     33%             36%             56%  ✓✓
ToxiGen:    4%              3%              6%   (still low)
```

Epoch 5 with higher threshold improves toxic recall across all datasets.

---

## 3. Validation Metrics Comparison

| Epoch | Val Tox F1 | Val Emo F1 | Combined |
|-------|------------|------------|----------|
| 3 | 0.7387 | 0.5618 | 0.6503 |
| 4 | 0.7526 | 0.5783 | 0.6655 |
| **5** | **0.7787** | **0.5825** | **0.6806** |

Epoch 5 has the best validation metrics overall.

---

## 4. Root Cause Analysis: Why Different from Partner A?

### Confirmed Differences

| Factor | Our Implementation | Partner A (Inferred) |
|--------|-------------------|---------------------|
| **Training paradigm** | Paired MTL (both tasks every batch) | Possibly interleaved single-task |
| **Pseudo-labels** | OLID→neutral, GoEmotions→non-toxic | Unknown |
| **Threshold** | Tuned (0.25-0.45) | Possibly 0.5 or tuned differently |
| **Epoch selection** | Tested 3, 4, 5 | Unknown |
| **GPU** | P100 | T4 (no impact on results) |

### Most Likely Causes (Ranked)

1. **Training Paradigm (High Impact)**
   - Our paired training with pseudo-labels introduces noise
   - Interleaved training avoids pseudo-label contamination
   - **Solution**: Implement interleaved training without pseudo-labels

2. **Threshold Difference (Medium Impact)**
   - Different thresholds affect absolute F1 significantly
   - Epoch 5 threshold (0.45) gives better L2/L3 than epoch 3/4 (0.25)
   - **Solution**: Already addressed via threshold tuning

3. **Data Preprocessing (Low-Medium Impact)**
   - Possible differences in text cleaning
   - **Solution**: Verify preprocessing is identical

4. **Random Seed (Low Impact)**
   - Different initialization can cause 2-3% variance
   - **Solution**: Fix random seed for reproducibility

---

## 5. Potential Improvements

### Quick Fixes (No Retraining)

| Fix | Expected Impact | Effort |
|-----|-----------------|--------|
| ✅ Threshold tuning | +3-5 pp on L2/L3 | Done |
| Test lower thresholds (0.15-0.20) | +1-2 pp on L3 | 5 min |
| Ensemble epochs 4+5 | +1-2 pp overall | 10 min |

### Requires Retraining

| Fix | Expected Impact | Effort |
|-----|-----------------|--------|
| Interleaved training (no pseudo-labels) | +5-10 pp | 1 hour |
| Better pseudo-labels (keyword-based) | +2-4 pp | 30 min |
| Train for more epochs (10) | +2-3 pp | 2 hours |
| Data augmentation (back-translation) | +3-5 pp | 3 hours |

---

## 6. Recommendations

### For Final Report

1. **Use Epoch 5 results** (best generalization)
2. **Report honestly** with methodology explanation
3. **Note that ΔF1 (0.28) is comparable to Partner A's (0.27)**
4. **Acknowledge limitations** of pseudo-labeling approach

### For Partner A Review

**Questions that still need answers:**

1. Did you use interleaved or joint training?
2. How did you handle samples without both labels?
3. What threshold was used for the stress test?
4. Which epoch was your final model?
5. What random seed was used?

---

## 7. Final Summary Table

| Metric | Partner A | Our Best | Gap | Assessment |
|--------|-----------|----------|-----|------------|
| L1 (OLID) | 0.7917 | 0.7096 | -8.2% | Significant |
| L2 (Jigsaw) | 0.7744 | 0.7346 | -4.0% | Moderate |
| L3 (ToxiGen) | 0.5168 | 0.4306 | -8.6% | Significant |
| **ΔF1** | **0.2749** | **0.2791** | **+0.4%** | **✅ Equivalent** |

### Key Insight

> While absolute F1 scores differ by 4-8%, the **cross-domain degradation pattern (ΔF1) is nearly identical**. This suggests both implementations capture similar generalization behavior, but with different baseline performance levels.

---

## 8. Artifacts for Download

1. `aura_epoch3.pth` - Epoch 3 checkpoint
2. `aura_epoch4.pth` - Epoch 4 checkpoint  
3. `aura_epoch5.pth` - **Best model (recommended)**
4. `aura_results.json` - Full results with history

---

**Author**: Partner B  
**Last Updated**: January 9, 2026, 23:30  
**Status**: Awaiting Partner A clarification on training paradigm
