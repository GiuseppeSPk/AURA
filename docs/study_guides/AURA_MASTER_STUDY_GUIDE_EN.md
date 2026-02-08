# AURA V10: MASTER TECHNICAL STUDY GUIDE

This document provides a comprehensive technical breakdown of the AURA V10 architecture, bridging the gap between fundamental Transformer theory and the specific multi-task innovations implemented in this project.

---

## 1. TRANSFORMER FOUNDATIONS & ATTENTION MECHANISMS

### 1.1 From Recurrence to Parallelism
Before the Transformer (Vaswani et al., 2017), sequences were processed by **RNNs** (LSTMs or **GRUs**).
- **Sequential Bottleneck**: RNNs process tokens one by one, leading to the "Vanishing Gradient" problem where long-range dependencies are lost.
- **The Transformer Advantage**: Unlike the GRU, which compresses history into a single hidden state, the Transformer utilizes **Self-Attention** to allow every word to "look" at every other word simultaneously.

### 1.2 Scaled Dot-Product Attention: The Latent Space Projection
AURA projects each token into three distinct **latent spaces** via learned weight matrices ($W_Q, W_K, W_V$):
1. **Query Space ($Q$)**: Represents "what I am looking for."
2. **Key Space ($K$)**: Represents "what information I offer."
3. **Value Space ($V$)**: Represents "the semantic content I carry."

**The Formula:**
$$ \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

> [!IMPORTANT]
> **Why Scaled?** In AURA, $d_k = 96$. Without dividing by $\sqrt{d_k}$, the dot products would grow too large, pushing the Softmax into regions with near-zero gradients (Saturation). Scaling ensures training stability.

---

## 2. ENCODER EVOLUTION: BERT, RoBERTa, AND DeBERTa

### 2.1 Why Encoders for AURA?
AURA is an **Encoder-only** system. Unlike Decoders (like GPT), which only see the past, Encoders are **fully bidirectional**. For classification tasks (Toxicity, Emotion), seeing the entire global context of a sentence is mandatory.

### 2.2 RoBERTa (Our Backbone)
AURA V10 uses **RoBERTa-base** due to its robust pre-training:
- **Dynamic Masking**: BERT used **Static Masking**, where the mask was applied once during data preprocessing; the model saw the same masked tokens in every epoch. RoBERTa generates masks on-the-fly, so the same sentence is masked differently every time it is seen, improving robustness and preventing pattern memorization.
- **No NSP**: Removing the Next Sentence Prediction task allows the model to focus purely on Masked Language Modeling (MLM).

### 2.3 DeBERTa: Disentangled Attention
In the project, we explore **DeBERTa-v3**, which separates content from position in the attention calculation. This "disentanglement" allows the model to understand syntax much better than standard models.

---

## 3. AURA V10 ARCHITECTURE: TS-MHA & POOLING

### 3.1 Task-Specific Multi-Head Attention (TS-MHA)
AURA introduces **Disentangled Feature Heads**. Instead of one shared attention block, we use 4 independent heads:
- **Tox-MHA, Emo-MHA, Sent-MHA, Rep-MHA**.
- **The Logic**: This prevents "Negative Interference." For example, the Reporting task focuses on verbs like "said," while the Toxicity task focuses on offensive adjectives. Separate heads allow them to specialize without mutual interference.

### 3.2 Mean Pooling with Masking
To convert token vectors into a single sentence vector, AURA uses **Mean Pooling**:
$$ \text{Output} = \frac{\sum (\text{Hidden States} \times \text{Mask})}{\sum \text{Mask}} $$
By using the attention mask, we ensure that **Padding tokens** (empty slots) do not pollute the final representation.

---

## 4. OPTIMIZATION: FOCAL LOSS & KENDALL UNCERTAINTY

### 4.1 Focal Loss: Tackling Class Imbalance
In Toxicity detection (95% non-toxic), standard Cross-Entropy is ineffective. AURA uses **Focal Loss**:
$$ FL(p_t) = -(1 - p_t)^\gamma \log(p_t) $$
The $(1-p_t)^\gamma$ term reduces the weight of "easy" (majority class) examples, forcing the model to focus on rare, difficult positive cases.

### 4.2 Kendall Uncertainty for Multi-Task Learning (MTL)
How do we balance 4 losses? AURA uses **Homoscedastic Uncertainty**:
$$ L_{total} = \sum \left( \frac{1}{2\sigma_i^2} L_i + \log \sigma_i \right) $$
The model **learns** the precision ($1/\sigma^2$) for each task. If a task is noisy (like Emotion), the model increases $\sigma_i$, effectively reducing that task's weight.

### 4.3 The Phantom Gradient Fix (V10.2)
**The Problem**: In batches missing a task (e.g., Sentiment), the $\log \sigma$ term remained in the graph, causing the model to learn fake confidence.
**The Fix**: AURA V10.2 uses a total mask. If a task is absent, the **entire term** (including the regularizer) is multiplied by zero, preventing "phantom" gradients from distorting the weights.

---

## 5. EVALUATION METRICS

### 5.1 The F1-Score vs. Accuracy
- **Accuracy Trap**: In imbalanced data, a model saying "Non-Toxic" 100% of the time gets 95% accuracy but is useless.
- **F1-Macro**: Used in AURA to give equal weight to rare classes, balancing **Precision** and **Recall**.
- **F1-Sample**: Used for the multi-label Emotion task to evaluate how well we hit the "set" of emotions per comment.

### 5.2 Bias Initialization
To stabilize the beginning of training, we initialize the output bias of the classification heads based on the dataset distribution (e.g., negative bias for rare toxicity). This prevents initial loss spikes and speeds up convergence.

---

## 🔬 TECHNICAL RESEARCH FAQ (POCKET GUIDE)

1. **Why 768 dimensions?** Balance between model capacity and computational efficiency; ensures head-divisibility (8 heads $\times$ 96-dim subspaces).
2. **Standard Cross-Entropy vs. Focal Loss?** Standard CE treats all classification errors equally; Focal Loss applies a modulating factor $(1-p_t)^\gamma$ to target rare, difficult examples (Toxicity minority class).
3. **Role of Key (K) vs Value (V)?** The Key is used for compatibility matching (contextual indexing); the Value carries the actual semantic information associated with that token.
4. **Why TS-MHA (Task-Specific Attention)?** To achieve feature disentanglement; allowing the model to focus on different linguistic markers (lexical for toxicity, eventive for reporting) without shared-representation interference.
