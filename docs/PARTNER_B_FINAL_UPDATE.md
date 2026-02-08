# 🦅 AURA Project - Partner B Final Update

Hey Partner, this is the definitive status update for the project handover.

We are currently at **Version 10 (V10)**. This version is architecturally complete and ready for training on Kaggle.

---

## 🛑 The Core Problem
Despite reaching decent F1 scores (~0.78 with V8), our previous models suffered from two critical qualitative flaws:

1.  **The "Traffic Paradox"**: The model confuses **Negative Sentiment** with **Toxicity**.
    *   *Example*: "I hate traffic" -> Predicted **TOXIC** (False Positive).
    *   *Why*: The shared encoder conflates "hate" (sentiment) with "abuse" (toxicity).

2.  **The Reporting Blindspot**: The model fails to distinguish between *being* toxic and *quoting* toxicity.
    *   *Example*: "He said 'you are stupid'" -> Predicted **TOXIC** (False Positive).
    *   *Why*: The model spots the word "stupid" but ignores the context verb "said".

---

## 💡 The Solution: AURA V10 "Scientific Standard"

We have rebuilt the architecture to specifically address these flaws using **Feature Disentanglement**.

### 1. Architecture: Parallel Multi-Head Attention
Instead of merging all tasks into one dense layer, V10 splits the processing into **4 Parallel Streams** immediately after the RoBERTa encoder.

*   **Logic**: Just like "Multi-Head Attention" in the course slides (Module 2), we use separate heads to look for different things.
    *   **Head 1 (Toxicity)**: Ignores "traffic", looks for "You" + Insults.
    *   **Head 2 (Reporting)**: Looks for "said", "reported".
    *   **Head 3 (Sentiment)**: Looks for "hate", "terrible".

By separating these, we hypothesize the "Traffic Paradox" will be solved because the Toxicity Head won't be distracted by Sentiment signals.

### 2. The Engine: Kendall Uncertainty Loss (Module 3)
You might ask: *"What happened to the Kendall Loss we discussed?"*
It is actively running in the background of V10.

*   **Role**: It acts as an **Automatic Mixer**.
*   **Problem**: The "Sentiment" task is huge (70k samples) and easy. The "Toxicity" task is small (12k) and hard.
*   **Solution**: Without Kendall, the model would only care about Sentiment because it generates the biggest total loss. Kendall's formula (`loss * exp(-sigma) + sigma`) automatically detects that Sentiment is "easy/noisy" and scales down its gradients, forcing the model to focus on the harder Toxicity task.

### 3. The "Data Engineering" Evolution (Full History)
This table explains exactly *why* we changed the data at each step, and what scientific method drove that decision.

| Phase | Dataset Composition | The Scientific Driver (Why?) | Specific Methodology |
|---|---|---|---|
| **V1-V7** | **OLID** (Toxicity) + **GoEmotions** (Emotion) | **Hypothesis: Emotional Proxy**. We believed toxicity could be detected via emotional signatures (Anger/Disgust). | *Multi-Task Learning* (Standard). |
| **V8** | **+ Hate Speech** (100k) | **Class Imbalance Problem** (Module 3). OLID had only ~800 toxic examples vs 58k emotion samples. The gradients were drowned out. | *Oversampling / Data Augmentation*. We injected external high-quality toxic samples to balance the P(y=1) prior. |
| **V8** | **+ SST-2** (Sentiment) | **Concept Disentanglement**. The model was failing on "I am sad" (predicting Toxic). It needed a "Negative Control Group". | *Contrastive Learning Logic*. By adding "General Negativity" (SST-2), we forced the model to learn that Negative $\neq$ Toxic. |
| **V10** | **+ Hard Negatives** (Custom) | **Adversarial Weakness**. Qualitative analysis showed the model failed on keywords like "Hate" in neutral contexts ("I hate rain"). | *Adversarial Training / Hard Negative Mining*. We manually created "Attacks" (sentences with trigger words but label 0) to robustify boundaries. |
| **V10** | **+ Reporting** (Custom) | **Error Analysis (Context)**. We noticed False Positives on journalists quoting hate speech. | *Contextual Data Injection*. We created a specific sub-dataset to teach the model the semantic role of "Quoting Verbs". |

**Key Takeaway**: We shifted from "Model-Centric" (trying to fix it with math/loss functions) to "Data-Centric" (fixing the input signal), which is the validated modern approach (Ng et al.).

---

## 📉 Status & Next Steps

### DONE ✅
*   **Architecture Verified**: Dry run passed locally. Dimensions are correct.
*   **Code Review**: The V10 notebook is fully compliant with Module 3 (Loss History plots, Confusion Matrix heatmaps are implemented).
*   **Data Prep**: The dataset `aura-v10-data` is ready in the `kaggle_upload` folder.

### TO DO 🔄
*   **Kaggle Training**: You need to upload the dataset and run the `AURA_V10_Kaggle.ipynb` notebook.
*   **Verify F1**: We expect the F1 to potentially surpass the V8 benchmark (0.78), but more importantly, we expect the **False Positive Rate** on the "Traffic" and "Reporting" test cases to drop significantly.

We have moved from a "Black Box" that just optimizes F1, to a "Glass Box" that is architected to understand context.
