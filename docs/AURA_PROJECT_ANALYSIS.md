# AURA Project: The Scientific Roadmap
**From Hypothesis to SOTA: A Narrative of Data & Robustness**

While V10 is technically robust, the volume of Reporting data was initially a bottleneck. In AURA V10.1, we reached the **500-sample "Sweet Spot"**, providing a much stronger signal for the citational head. The next evolution focuses on:
This document outlines the evolutionary path of Project AURA. It is not just a list of models, but a chronological story of scientific inquiry, failures, and the data-driven decisions that led to the final V8 solution.

---

## 🧭 The Journey at a Glance

Our goal was never just "High F1". It was **Robustness**.
We moved through three distinct eras:
1.  **The Baseline Era (V1-V3)**: Establishing a working prototype (F1 ~0.72).
2.  **The "Loss Engineering" Era (V4-V7)**: Trying to solve problems with complex math (F1 ~0.78, but unstable).
3.  **The "Data Engineering" Era (V8-V9)**: Solving problems with better data (F1 ~0.77, Stable).
4.  **The "Architecture Refinement" Era (V10 - FINAL)**: Disentangling features via Task-Specific Attention.

---

## 1. Phase 1: The Foundation (V1 - V3)
**Hypothesis**: *"Multi-Task Learning with simple Class Weights can beat single-task BERT by learning to detect Toxicity via Emotions."*

### V3: The First Competitive Baseline
We established our first working model using `bert-base-uncased` and simple Class Weights (Inverse Frequency) to handle the 1:2 imbalance in OLID.
*   **Result**: F1 **~0.72**.
*   **Scientific Verdict**: The model worked but was biased. Qualitative analysis showed it was over-predicting "Neutral" emotions because the sheer volume of neutral samples dominated the gradients.
*   **Action**: We needed a dynamic way to balance tasks, not just static weights.

---

## 2. Phase 2: The "Loss Engineering" Trap (V4 - V7)
**Hypothesis**: *"We can fix the imbalance and overfitting using advanced mathematics (Focal Loss + Bayesian Uncertainty) without adding more data."*

This phase was crucial. We explored the limits of **algorithmic optimization**.

### V4: The Mathematical Peak (Jan 17 2026)
We implemented the state-of-the-art **Kendall Uncertainty Loss** ($\sigma$ weighting) + **Focal Loss** ($\gamma=2.0$).
*   **Result**: F1 hit a record **0.7799**.
*   **The Hidden Failure**:
    *   Training F1: **0.9500** (Near perfect memorization).
    *   Validation F1: **0.7799**.
    *   **The Gap**: **18.4%**.
*   **Diagnosis**: The model effectively "hacked" the loss function. It memorized the training set noise to minimize the uncertainty term, failing to generalize. This proved a critical ML lesson: **You cannot math your way out of data starvation.**

### Intermediate Experiments (V5-V7: Ablation Studies)
To confirm V4 wasn't a fluke, we ran ablation studies to reduce overfitting:
*   **V5 (DistilBERT)**: Switch to smaller backbone (66M params). *Result: Underfitting (F1 ~0.74).*
*   **V6 (High Dropout)**: Increased dropout to 0.5. *Result: Slower convergence, but gap remained >10%.*
*   **V7 (Aggressive Decay)**: Heavy weight decay. *Result: Loss of precision.*

**Conclusion of Phase 2**: We hit a ceiling. The architecture was fine, but the *signal* was too weak compared to the noise.

### Advanced Experimental Branches (The "Overfitting" Wars)
To ensure we weren't just missing a trick, we launched two sophisticated experimental protocols:

#### **V11: Project NEMESIS (Adversarial Training)**
*   **Hypothesis**: The model is overfitting because it uses "Negative Sentiment" as a cheat code for "Toxicity".
*   **Method**: Added a **Gradient Reversal Layer (GRL)**. An "Adversary Head" tries to guess if a sample is from the Toxicity or Sentiment dataset. The encoder tries to *fool* this head (maximize error).
*   **Outcome**: Technically fascinating, but unstable training dynamics. The model struggled to converge, confirming that "Loss Engineering" was becoming too complex for the available data.

#### **V12: Project OMEGA (Soft Labels)**
*   **Hypothesis**: Overfitting comes from "Hard Negatives" (e.g., "I hate Mondays" labeled as 0). Forcing the model to predict 0.00 is wrong; it should be allowed to be uncertain.
*   **Method**: **Label Smoothing** (soft targets) + **Bias Initialization** (starting training assuming 95% non-toxic).
*   **Outcome**: Improved stability, but F1 remained around 0.77. 

These sophisticated attempts confirmed a hard truth: **We didn't need better gradients, we needed better examples.** This led directly to Phase 3.

---

## 3. Phase 3: The "Data Engineering" Solution (V8 - Final)
**Hypothesis**: *"The only way to close the Generalization Gap is to expand the semantic signal itself. We need a massive, balanced Multi-Head ecosystem."*

### V8: The "Ferrari with Fuel" (Jan 18 2026)
We abandoned the "small data + complex math" approach. We moved to a **Data-Centric strategy**:
1.  **Mega-Dataset**: Expanded to **113k samples**.
2.  **4-Task Architecture**: Added **Sentiment** (SST-2) and **Hate Speech** explicit tasks.
3.  **Task Balancing**: Critically, we *capped* the dominant Sentiment task to 15k samples to match Toxicity/Emotion.

### The Result (Gold Run)
*   **Training Dynamics**:
    *   Epoch 1: F1 0.72 (Warmup)
    *   Epoch 2: F1 0.76 (Learning)
    *   Epoch 4: F1 **0.78** (Stable SOTA)
*   **The Defining Victory**:
    *   V4 Gap: **18.4%** (Dangerous Overkill)
    *   V8 Gap: **3.2%** (Healthy Learning)

### Final Verdict for the Defense
Why is V8 the winner if F1 is similar to V4?
> "V4 was a Fragile Sprinter. V8 is a Marathon Runner."
V8 achieved the same high performance (0.78) but did so **honestly**, by learning robust features across 4 tasks, whereas V4 essentially memorized the exam questions. V8 is the only model we would trust in production.

---

## 4. Phase 4: The Stability & Integrity Revolution (V10.2 - FINAL)
**Hypothesis**: *"Even the best architecture fails if the loss dynamics are unstable or the validation data is contaminated."*

In late Jan 2026, we performed a rigorous **Pre-Defense Audit** of AURA V10.1, leading to the definitive **V10.2 Gold Standard**.

### 1. The "Phantom Gradient" Fix
We discovered a critical mathematical flaw in the multi-task loss implementation. When a task was absent, its regularization term was still pulling gradients, causing "uncertainty collapse" and exploding weights.
*   **Solution**: Implemented a **Task Masking Layer** in `UncertaintyLoss`.
*   **Scientific Result**: Perfectly stable training across 15+ epochs.

### 2. The Data Integrity Audit (Reporting Task)
We detected "Template Leakage" in the Reporting task. Similar phrases with minor typos were split between train and validation, creating an artificial F1 of 1.0.
*   **Solution**: Robust deduplication (removing ~485 near-duplicates) and a clean 90/10 stratified split.
*   **Scientific Result**: A realistic, honest F1 (~0.90), proving the model actually *learns* the reporting feature rather than memorizing templates.

---

## 5. Critical Reflection: The Architecture "Pivot"
**(A specific answer for the Exam Jury)**

A key question might be: *"Multi-Head Attention is standard (Module 2). Why didn't you use Task-Specific Attention immediately in V3/V4?"*

**Our Initial Rationale (The "Parsimony" Argument)**:
We initially rejected adding custom Attention layers based on the **Occam's Razor** principle:
1.  **Redundancy**: BERT-base already contains **144 internal attention heads** (12 layers $\times$ 12 heads). We hypothesized that the pre-trained `[CLS]` token was sufficiently rich to encode all necessary features without adding architectural complexity.
2.  **Overfitting Risk**: Our primary enemy was the small dataset size (OLID ~13k). Adding full $W_q, W_k, W_v$ matrices for 4 separate tasks would drastically increase the parameter count, likely accelerating the overfitting we were trying to avoid. We opted for a "lightweight" shared-encoder design.

*   This realization led to the **V10 Final design**: accepting the parameter cost of Attention Heads was necessary to achieve **Feature Disentanglement** and acknowledge the **Perspectivist NLP** (Ref: **V. Basile**) nature of the task.

---

## 5. Technical Deep Dive: The Hidden Mechanics
**(Essential "Under-the-Hood" Details for the Defense)**

While the architecture gets the glory, the **training stability** comes from these often-overlooked engineering decisions:

### 5.1 The Loss Function Ecosystem
We don't just "calculate loss"; we orchestrate it based on task nature:
*   **Binary Cross Entropy (BCE)**: Used for **Emotion** and **Reporting**.
    *   *Why?* **Emotion** is Multi-Label (can be both *Joy* and *Surprise*). Softmax would force a single choice; BCE allows independent probabilities. **Reporting** is a clean binary signal where valid samples are abundant.
*   **Focal Loss ($\gamma=2.0$)**: Used for **Toxicity** and **Sentiment**.
    *   *Why?* These tasks suffer from extreme class imbalance (95% Non-Toxic). Standard BCE would let the model get lazy (predicting 0 everywhere). Focal Loss mathematically "down-weights" easy examples, forcing the model to learn from the rare, hard toxic cases.
*   **Kendall Uncertainty (Softplus)**: The "Manager" of the losses.
    *   *Why Softplus?* In V10, we replaced the exponential formulation $e^s$ with $\text{Softplus}(s)$. This prevents numerical explosions (NaNs) if the model becomes too uncertain, ensuring the variance $\sigma^2$ stays positive but stable.

### 5.2 Tokenization Decisions
*   **RoBERTa BPE (Byte-Pair Encoding)** vs BERT WordPiece.
    *   *Why it matters*: Toxicity often involves obfuscation (*"fvck", "sh!t"*). BERT's WordPiece breaks these into meaningless chunks (`[UNK]`). RoBERTa's BPE operates at the **byte level**, allowing it to learn that *"f@@k"* is semantically close to *"fuck"* without needing a dictionary update.

### 5.3 Training Dynamics
*   **Gradient Accumulation (Steps=4)**:
    *   *The Logic*: We train with Batch Size 16 (to fit in GPU memory) but accumulate gradients for 4 steps before updating. This simulates a **Virtual Batch Size of 64**. Larger batches provide smoother, more accurate gradient estimates, reducing training noise.
*   **Mixed Precision (FP16)**:
    *   We use 16-bit floating point for the forward pass. This doubles our available memory and speeds up training by ~40% without losing predictive accuracy.
*   **Linear Warmup (10%)**:
    *   We don't start at full learning rate. We ramp up linearly for the first 10% of steps. This allows the AdamW optimizer to calculate correct momentum statistics before making large updates, preventing early divergence.

### 5.4 Initialization Secrets
*   **Bias Initialization (Log-Odds)**:
    *   *The Hack*: Toxicity is 5% positive. A standard model guesses 50/50 at iteration 0, getting a huge initial loss. We manually set the final layer bias $b = \log(0.05/0.95) \approx -2.94$.
    *   *The Result*: The model **starts** by predicting 5% probability. This skips the first epoch of "learning to just say no" and allows immediate feature learning.
*   **Label Smoothing (0.1)**:
    *   We never tell the model "This is 100% Toxic". We say "This is 90% Toxic". This prevents the model from becoming overconfident and makes the decision boundary more robust to noise.

### 5.5 Input Engineering & Inference
*   **The 128-Token Decision**:
    *   We truncated input to 128 tokens. Analysis showed 99% of social media insults occur in the first 50 words. Extending to 512 would quadruple memory usage ($O(N^2)$ attention cost) with zero accuracy gain.
*   **Special Tokens**:
    *   Input is formatted as `<s> text </s>`. We use the embedding of the `<s>` (CLS) token as the pooled representation passed to the 4 task heads, as it aggregates the entire sequence context.
*   **Inference Thresholding**:
    *   **Toxicity/Report/Sentiment**: Argmax (0 vs 1).
    *   **Emotion**: Independent Thresholding at 0.5 (Sigmoid). This allows the model to predict *multiple* emotions (e.g., *Anger + Disgust*) or *no* strong emotion, reflecting real human complexity.

---

### 6. Alignment with Course Modules (The "Tick-Box" Strategy)
**For the Exam Jury: Proof of Curriculum Integration**

We explicitly designed AURA to satisfy the requirements of **Module 3 (Prof. Requirements)**:

| Technique | Course Reference | AURA Implementation |
| :--- | :--- | :--- |
| **Class Weighting** | `11 - Imbalanced datasets` | Implemented Inverse Class Frequency weights in V3. |
| **Focal Loss** | `11 - Imbalanced datasets` | Applied with $\gamma=2.0$ in V4 and V8. |
| **Hierarchical Loss** | **Scientific Novelty** | **Micro (Focal) + Macro (Kendall) Optimization Loop.** |
| **Uncertainty Weighting** | `12 - Multi-task Learning` | Kendall et al. (2018) dynamic weighting. |
| **Bias Initialization** | `11 - Imbalanced datasets` | Used `log(pos/neg)` init for the Toxicity Head to speed up convergence. |
| **Early Stopping** | `10 - Overfitting` | Strict `patience=3` on Val F1 to prevent memorization. |
| **Dropout** | `10 - Overfitting` | Tuned to 0.4 (High) in V8 to force robust feature learning. |
| **Data Augmentation** | `11 - Imbalanced datasets` | The "Hate Speech" and "Sentiment" tasks act as semantic augmentation. |
| **Perspectivism** | `Modulo 2 - Ethics` | Implementation of Basile's theory via Multi-Head subjectivity analysis. |
| **Event Annotation** | `Sprugnoli (Theory)` | Use of POS eventive shades for Reporting Head (Ref: Sprugnoli). |

---

## 7. Summary of Evolution

| Model | Technique | F1 Score | Overfit Gap | Status |
| :--- | :--- | :--- | :--- | :--- |
| **V3** | Static Weights | 0.72 | Low | **Baseline** (Functional) |
| **V4** | Focal + Uncertainty | **0.78** | **18.4%** | **Rejected** (Overfitting) |
| **V8-V9** | Data Balancing | 0.77 | 3.2% | **Milestone** |
| **V10.2**| **Task Attention + Masked Loss**| **TBD** | **~2%** | **FINAL (GOLD STAND.)** |

---
*Roadmap finalized for Defense preparation: Jan 2026.*
