# AURA: Progress Report

## 1. Project Overview
**Goal**: Evaluate and improve **Domain Robustness** in toxicity detection by mitigating the performance drop caused by context shift (e.g., Twitter-to-Wikipedia).
**Method**: **Task-Specific Multi-Head Attention** for Domain-Invariant Feature Learning.
Unlike standard Multi-Task Learning, which may conflate features, we use 4 parallel attention mechanisms to isolate universal affective signals from domain-specific vocabulary.

## 2. Evolution Path

### Phase 1: Native Domain Benchmarking (V1-V9)
Initially, we established performance benchmarks using standard architectures (**BERT-base**, **MTL with Shared Pooling**).
**The Result**: We achieved stable performance (**F1 ~0.74**) within the training domain. 
**The Research Observation**: Upon closer evaluation of the model's behavior during context shifts, we observed a reliance on lexical triggers. This provided the empirical motivation to test more complex architectures that could isolate intent from vocabulary.

### Phase 2: Testing the Disentanglement Hypothesis (V10 Final)
To evaluate if multi-task learning could truly provide domain-invariant features, we transitioned to **V10**, which implements a **Task-Specific Attention** mechanism. This allows us to test whether giving the model specialized "eyes" for different signals improves its ability to generalize across domains.

1.  **Head A (Sentiment)**: Focuses on lexical polarity (e.g., "hate", "nice").
2.  **Head B (Toxicity)**: Focuses on intentional harm and targeted insults.
3.  **Head C (Reporting)**: Detects quotation markers ("he said") to neutralize reported toxicity.
4.  **Head D (Emotion)**: Focuses on universal affective triggers (Anger/Disgust).

By isolating the **affective signature** (Head D) from the **lexical triggers** (Head B), we leverage the fact that emotions are domain-invariant markers. This allows the model to maintain stability even when the input vocabulary shifts (e.g., from Twitter slang to Wikipedia formalisms).

## 3. Methodological Details

### 3.1 Data Strategy
We aggregated 4 datasets into a unified pipeline:
*   **Toxicity (Training)**: OLID (~12k samples). Imbalanced at **33%** toxic classes.
*   **Emotion (Training)**: GoEmotions (~57k samples). Severe imbalance in Disgust (7%) and Fear (5%).
*   **Sentiment (Training)**: SST-2 (~72k samples). Balanced binary sentiment.
*   **Cross-Domain Evaluation**: Jigsaw (Wikipedia) and ToxiGen (Synthetic) are used exclusively for testing **Domain Robustness**.
*   **Methodology**: Imbalance is handled via **Weighted Focal Loss** (`[0.5, 2.0]` weights) rather than dataset oversampling, to preserve data integrity.
*   **Data Augmentation & Robustness**: We specifically integrated two types of "Stress Data" into the pipeline:
    *   **Reporting Examples (4th Task)**: A dedicated supervised dataset (**~1,000 unique samples** after deduplication) of "Direct Insults" vs. "Reported Speech" (e.g., *"He said you are X"*). This provides the explicit signal for the **Reporting Attention Head**.
    *   **Hard Negatives**: We ensured the training stream includes "aggressive but non-toxic" samples (e.g., heavy sentiment adjectives from SST-2 and frustrations from OLID) to force the **Toxicity Head** to ignore lexical triggers and wait for a "Targeted Insult" signal from its task-specific attention.

### 3.2 Loss Function (Kendall et al. & Focal Loss)
We implement a hybrid loss strategy to handle both task-balancing and class-imbalance. A notable architectural choice is our use of **Homoscedastic Uncertainty Weighting** (Kendall et al., 2018).

**Why Kendall?**
Although not explicitly covered in the course modules, we chose this approach after identifying a bottleneck: manually tuning static loss weights ($\lambda_i$) for 4 heterogeneous tasks (Binary, Multi-label, Sentiment) proved highly unstable. Kendall's method allows the model to learn a dynamic scalar $\sigma_i$ for each task loss $L_i$, acting as a principled "parameter-free" balancer. 

$$ \mathcal{L}_{total} = \sum \text{mask}_i \cdot \left( \frac{1}{\text{softplus}(\sigma_i^2)} \mathcal{L}_i^{focal} + 0.5 \cdot \text{softplus}(\sigma_i^2) \right) $$

**V10.2 Refinement**: In the final version, we introduced **Task Masking** and **Softplus Regularization**. This ensures that absent tasks do not contribute "phantom gradients" to the uncertainty parameters, preventing the precision weights from exploding on sparse tasks.

## 4. Crucial Questions for Faculty

**Q1: Attention Mechanism Validity**
Is our implementation of "Task-Specific Attention" (separate $W_Q, W_K, W_V$ projections per task applied to the *same* encoder output) theoretically sound for feature disentanglement, or should we have used separate encoders?

**Q2: Interaction with Contextual Features**
We are investigating **Reporting Detection** (identifying quotation markers) as an auxiliary task to improve context-awareness. Is it methodologically more robust to treat this as a supervised secondary task, or should we consider an adversarial approach (Gradient Reversal) to decorrelate the toxicity head from reported-speech lexical patterns?

**Q3: Task Interference and Imbalance**
Our primary task (Toxicity) is natively imbalanced (~33% toxic). To preserve the natural distribution, we avoid oversampling and instead use **Weighted Focal Loss** (`[0.5, 2.0]` weights). In the auxiliary tasks, some classes are as rare as 5%. Is there a risk that the Kendall Uncertainty weighting might penalize these rare-class tasks too heavily because they appear 'more noisy' during early training, or is this dynamic balancing sufficient?

---

## 📚 References

1.  **Kendall, A., Gal, Y., & Cipolla, R. (2018)**. *"Multi-task learning using uncertainty to weigh losses for scene geometry and semantics."* Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR).
2.  **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017)**. *"Focal loss for dense object detection."* Proceedings of the IEEE international conference on computer vision (ICCV).
3.  **Liu, Y., et al. (2019)**. *"RoBERTa: A robustly optimized BERT pretraining approach."* arXiv preprint arXiv:1907.11692.
