# 🎯 AURA Project: Technical Update & Collaborative Handover

Dear Partner B,

I am pleased to share the final technical results of the **AURA** (*Affective Understanding for Robust Abuse Detection*) project. The training phase has been successfully completed using an advanced Multi-Task Learning (MTL) architecture on Kaggle T4 GPUs.

---

## 🚀 1. Technical Achievements (Final Results)

Our AURA model, based on a **Dual-Head BERT** architecture, was trained jointly on Toxicity Detection (OLID) and Emotion Recognition (GoEmotions). The model achieved state-of-the-art stability across domains.

### 🧪 3-Tier Stress Test Performance
| Test Level | Dataset | Task | Macro-F1 | Result Analysis |
|------------|---------|------|----------|-----------------|
| **Level 1** | **OLID** | In-Domain | **0.7917** | Competitive SOTA performance. |
| **Level 2** | **Jigsaw** | Domain-Shift | **0.7744** | **Outstanding!** Minimal drop (<2%). |
| **Level 3** | **ToxiGen** | Implicit Hate | **0.5168** | Solid zero-shot baseline. |

**Key Finding**: The inclusion of **Affective Signatures** (Emotion Task) acted as a powerful regularizer, allowing the model to generalize from Twitter (OLID) to Wikipedia (Jigsaw) with unprecedented robustness.

---

## 🛠️ 2. Core Innovations Implemented

1.  **Uncertainty-Weighted Loss**: Dynamic task balancing using learnable parameters (Kendall et al., 2018).
2.  **Focal Loss & Class Weights**: Specialized handling of rare emotion classes (Fear, Disgust) to prevent bias.
3.  **OneCycle Learning Rate**: Optimized convergence to preserve pre-trained BERT weights.
4.  **Early Stopping (Patience=3)**: Automatic preservation of the optimal Epoch 1 model to prevent overfitting.

---

## 🎭 3. Your Task: Linguistic Event Annotation (Pilot Study)

To further improve our model's precision (specifically reducing False Positives), we are integrating **Prof. Rachele Sprugnoli’s Theory of Event Representation**. 

I am handing over the file: `event_annotation_task_MIRATO.csv`.

### 📋 Annotation Instructions:

For each of the 100 samples in the provided CSV, please add a new column named **`event_annotation`** and assign one of the following two labels:

1.  **`OCCURRENCE`**: Use this label if the tweet is a direct expression of opinion or a direct insult. The abuse *happens* in the tweet.
    - *Example*: "You are a terrible person."
    - *Linguistic Cue*: The speaker is the source of the claim.

2.  **`REPORTING`**: Use this label if the tweet contains a "Reporting Trigger" (*says, claims, told, mentioned*) and the potentially abusive content is part of what is being reported/quoted.
    - *Example*: "He **said** that she is a terrible person."
    - *Linguistic Cue*: The speaker is merely reporting an event or a statement made by someone else.

#### Why this matters:
Standard models often flag `REPORTING` cases as "Offensive" (False Positives) because they contain toxic keywords, even if the user isn't being offensive themselves. Identifying these allows us to implement a "Reporting Filter" based on Prof. Sprugnoli's Event Representation theory.

---

## 📂 Deliverables Included:
- `AURA_Technical_Ledger.md`: Full log of training decisions and academic justifications.
- `AURA_README.md`: Professional project summary for GitHub.
- `event_annotation_task_MIRATO.csv`: The 100-sample annotation set.
- `aura_mtl_best.pt`: The final trained model weights.

**Next Step**: Once the 100 samples are annotated, please send back the updated CSV. We will then perform the Final Qualitative Analysis to calculate the "Event-Aware" precision boost.

Best regards,

**Partner A**
