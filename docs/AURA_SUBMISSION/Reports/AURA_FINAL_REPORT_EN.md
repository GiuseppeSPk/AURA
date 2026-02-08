# AURA: Final Technical Report
**Technologies for Multimodal Data Representation and Archives (2025/2026)**

**Team**: Giuseppe Spicchiarello, Mahmoud Hosseini Pour  
**Date**: January 28, 2026  
**Final Version**: V10.2

---

## 1. An Honest Introduction

When we started this project, the objective was straightforward: *"Let's see if BERT can detect toxicity."*

We thought we'd do a simple fine-tuning on OLID and call it done. However, looking at the initial results, we noticed a problem: the model couldn't distinguish *toxicity* from the use of *vulgar terms*. Phrases like *"Questo film fa cagare"* (This movie sucks) were flagged as toxic, when in reality they're just vulgar but not offensive toward a person.

This led us on a journey far more complex than expected, teaching us a fundamental lesson: **high numbers mean nothing if the model cheats.**  
We discovered that even the most advanced models are genuinely lazy—the first pattern they identify is the one that requires the least effort to minimize error, and if you're not careful, you risk obtaining apparently solid results that aren't actually aligned with the task.

This document tells the complete story of our work: the failed attempts, the nights spent debugging, and the architectural choices that led us to the final model.

---

## 2. The Initial Problem

### 2.1 What Doesn't Work in Standard Models?

Current toxicity detection models suffer from what we've called **"Domain Brittleness"**. In practice:
- A model trained (or fine-tuned) on Twitter collapses when tested on Wikipedia.
- A model that sees the word "idiot" always classifies it as toxic, even in sentences like *"He said I'm an idiot"* (which is a report, not a direct insult).

The fundamental problem is that these models learn **statistical associations**, not **intentions**.

### 2.2 Our Hypothesis (AURA)

We asked ourselves: *"Is there something more stable than words for identifying toxicity?"*

The answer came from studying two concepts:
1. **Perspectivism (Prof. Basile)**: The need to preserve personal and subjective nuances of annotators is fundamentally important when analyzing emotions. Toxicity is subjective. Two annotators may disagree. Instead of forcing a binary label, we can use *emotions* (particularly anger and disgust) as a more universal proxy.
2. **Inspiration from event linguistics**: Reading *"La rappresentazione linguistica degli eventi e la loro annotazione nei testi"* (Sprugnoli, 2024), the observation that events can be classified based on their nature and function inspired one of the project's objectives: identifying reporting cases to declassify them as non-toxic. A toxic phrase *directly stated* is a very different event from a toxic phrase *reported*. This insight led us to create the "Reporting Detection" task as the model's fourth head.
**Disclaimer**: The contents of Professor Sprugnoli's book were solely a source of inspiration for the project, they were neither plagiarized nor used except as free inspiration for academic research purposes. (We thank the professor for her valuable contribution).

**The AURA hypothesis**: By combining emotion analysis with awareness of linguistic framing, we can create a more stable and robust model.

---

## 3. Our Journey (From V1 to V10.2)

### 3.1 Phase 1: The Baseline and Initial Doubts (V1-V2)

We started with the most straightforward approach possible: a `bert-base-uncased` model with a single classification head (**V1**). We chose the **OLID** (Offensive Language Identification Dataset) because it's the industry "gold standard." Technically, it's a corpus of **14,100 tweets** hierarchically annotated (Level A: offensive vs. non-offensive language; Level B: targeted vs. generic offense; Level C: individual, group, or other target). It's a consolidated benchmark and represented the perfect starting point to understand whether BERT could distinguish a real insult from a simple colloquial phrase.

The initial idea was: *"Let's fine-tune on OLID and we're done."*

**V1 Result**: F1 **~0.72**.
It seemed perfect! Job done! But, driven by healthy student skepticism, we ran manual tests (qualitative stress tests with vulgar but non-offensive phrases) and something didn't convince us at all. We were right to doubt.

**The Pivot to Multi-Task (V2)**:
The V1 test revealed that the model was "lazy": it didn't distinguish true toxicity from the use of vulgar terms. A phrase like *"Cazzo, che freddo che fa oggi!"* (Damn, it's cold today!) was flagged as offensive just as much as a personal attack. This is why we moved to **V2**, integrating the **GoEmotions** dataset. This is where our journey toward **Multi-Task Learning (MTL)** begins. The hypothesis was that by teaching the model to recognize anger and disgust, we would "educate" it to ignore simple vulgarity and focus on aggressive intent.

**Problems encountered**:
1. **Statistical "Laziness" and the Neutral Trap**: The model immediately showed a tendency to choose the path of least resistance. For the emotion task, we used the **GoEmotions** dataset (HuggingFace), which originally has 27 categories. To make the signal stronger, we mapped these categories onto **Ekman's** taxonomy (6 basic emotions), but added a **seventh explicit class for "Neutral"**.
   - **The problem**: About 30% of samples were "Neutral". The model learned that, when in doubt, predicting "Neutral" drastically lowered the loss without having to make the effort to distinguish between Anger and Disgust. This led to "inflated" accuracy but zero generalization capability (overfitting).
2. **Signal Integrity (No Pseudo-Labels)**: Initially, we hypothesized "coupling" the tasks by automatically assigning a "Non-Toxic" label to all GoEmotions samples (pseudo-labeling). However, we realized that GoEmotions (extracted from Reddit) contains emotionally charged phrases that are *intrinsically toxic*. Forcing the model to see them as safe (`tox=0`) introduced a lot of noise, "confusing" the decision boundary. We therefore chose an **Interleaved Single-Task Batches** approach: each batch contains samples of only one type (e.g., only OLID or only GoEmotions). Missing tasks are marked with an `ignore_index` value (-1), so that the toxicity gradient isn't minimally influenced by emotional data noise and vice versa. This "gradient purity" was key to stabilizing training, taking us from an F1 of 0.71 to about 0.73-0.74 in early intermediate versions, but most importantly we wanted the model to learn concepts and not just statistical noise.

### V3: The Beginning of the Multi-Task Era
With **V3** we consolidated the setup: `bert-base-uncased` with parallel heads for Toxicity and Emotion, using static weights (Inverse Frequency) to balance classes.

**V3 Result**: F1 **~0.72**.

Although we had surpassed the baseline, qualitative analysis gave us a cold shock: the model was heavily "biased". It almost always predicted the "Neutral" emotion because the volume of neutral data dominated the gradients, and static weights couldn't give enough importance to rare emotions like Anger and Disgust.

We were at a crossroads: add more data (which we didn't have yet) or try to "force" the model with mathematics. We chose the second path.

---

### 3.2 Phase 2: The Mathematical Trap (V4-V7)

Frustrated by V3's "limitations," we thought the solution was purely algorithmic. If static weights weren't enough, we'd use dynamic and intelligent weights. We implemented an advanced mathematical arsenal:
- **Focal Loss** (Lin et al., 2017): Seeking a valid solution among course materials, particularly in the lessons dedicated to imbalanced datasets in **Module 3 (Prof. Bioglio)**, we identified this loss function as the potential key. Instead of using static class weights, which weight the entire class a priori, we implemented Focal Loss to dynamically weight individual examples. We set the **focusing parameter $\gamma=2.0$**, the "gold standard" value to drastically reduce loss on "easy" examples (those the model classifies with high confidence) and force the encoder to focus only on **Hard Negatives**. This approach is much more refined than simple weighting, because it allows the model not to waste gradients on already-learned patterns (like explicit toxicity) and focus on ambiguities.
  
  **The Focal Loss limitation**: We soon realized that, although Focal Loss solved the *internal* imbalance of each task (Easy vs Hard), it didn't resolve the "war" between the tasks themselves. In our Multi-Task system, the gradients of emotions and toxicity "fight" for precedence. Trying to balance them manually (e.g., *"weight toxicity 1.0 and emotion 0.5"*) was a nightmare of trial and error that never led to stable equilibrium. Therefore, we sought a mathematical solution to circumvent the problem.

- **Kendall Uncertainty Loss** (Kendall et al., 2018): This paper would become our methodological "salvation" for managing dynamic task balancing and, especially, for **managing intrinsic data noise**. Instead of seeking fixed weights, we treated Multi-Task Learning as a **Bayesian inference** problem, implementing **homoscedastic uncertainty** ($\sigma$) as a learnable parameter for each head.
  
  **The mathematical logic**: For each task $i$, the loss function becomes $\mathcal{L}_{total} = \sum \frac{1}{2\sigma_i^2}\mathcal{L}_i + \log\sigma_i$.
  - **Noise and Imbalance Management**: This formula was fundamental for the emotion task. In our dataset (V10.2), out of over 57,000 samples, **Neutral** dominates with **31,446** occurrences, while **Anger** (**17,771**) and especially **Disgust** (**4,053**) are decidedly rarer and carry strong annotation noise (the subjectivity of perspectivism). The term $\sigma_i^2$ allows the model to "absorb" this noise: if a task is too chaotic or imbalanced in a batch, the model increases the uncertainty $\sigma_i$, "lowering the volume" of that task to avoid ruining the encoder's features, while still allowing learning of clearer patterns.

  **Our implementation (V10.2)**: Compared to the paper's standard version, we introduced a modification for numerical stability. Instead of using the exponential function for $\sigma$ (which led to gradient explosions during RoBERTa fine-tuning), we used **Softplus** to derive the variance. This guarantees that the weight is always positive and that the transition during uncertainty learning is smooth, allowing the "still immature" task (Reporting) not to destroy consolidated performance on the main task (Toxicity).

**V4 Result**: We achieved a peak **Validation F1 of 0.78**. Initially we celebrated: it seemed the power of Kendall Loss and Focal Loss had finally unlocked the model's performance.

**V4 Disappointment (The F1 Illusion)**:
Despite initial celebrations, we realized something was wrong. While the validation set showed 0.78, on the training set the model reached **0.95**. It was a very clear signal: the model wasn't learning to understand language, it was simply **memorizing** the data.

We understood that Kendall Loss's flexibility had become a double-edged sword: the model used it to "hide" its errors by increasing uncertainty on difficult cases, ending up learning phrases by heart instead of generalizing. If the phrase wasn't identical to one already seen, the model failed.

**Failed attempts (V5-V7)**: before understanding that the problem was in the data, we spent a lot of time tinkering with parameters:
- **V5**: We tried **DistilBERT** to have a smaller model less prone to memorization, but we only got poor performance.
- **V6**: We raised **Dropout to 50%**, but training became very slow and unstable. Too many deactivated neurons prevented the model from capturing even the simplest patterns.
- **V7**: We tried very aggressive **Weight Decay**, but the model started "forgetting" even the right things.

We were at a dead end. Mathematics couldn't support a crumbling structure, the only plausible option left was **data engineering**.

---

### 3.3 Phase 3: Back to Reality (V8-V9)

With the awareness that no algorithmic optimization would solve structural problems, we shifted focus: less parameter "tuning" and more care in sample selection. Phase 3 was our **data-centric revolution**.

**V8 Strategy (Data Engineering)**:
We decided to balance tasks not just with loss, but at the root.
1. **Deduplication (The Unconvincing F1 Problem)**: Even though validation numbers were good, when we tested the model with phrases we wrote ourselves, it often failed. We asked ourselves: "If F1 is high, why does the model seem so stupid?" Checking the files more carefully, we found the answer: there were duplicates between training and validation sets. The model wasn't guessing, it was just remembering answers it had already seen. Cleaning the data was the first step to having an honest evaluation of the model.
2. **Data Bottleneck (Class Imbalance)**: Analyzing the composition, we realized the imbalance: **OLID** (~12,000 samples for toxicity) was "drowned" by **GoEmotions** (~57,000 samples for emotions). The main task's gradients were in the minority.
   
3. **SST-2 as "Semantic Anchor"**: Error analysis revealed a critical pattern: the model confused generic negativity with toxicity. Phrases like *"I hate traffic"* were classified as toxic. To solve this, we integrated the **SST-2** dataset (Stanford Sentiment Treebank, ~72,000 samples). The goal was to teach the model that **Negativity $\neq$ Toxicity**: SST-2 contains thousands of negative movie reviews, where negativity is directed at objects and not people.
4. **Auxiliary Task Capping**: Despite the utility of SST-2 and GoEmotions, their volume (over 130,000 total samples) risked "drowning" the Toxicity signal. We therefore applied **limited sampling**: SST-2 and GoEmotions were capped at **20,000 samples each**, while Toxicity remained intact (~12,000). This way, each task's gradients contributed in a balanced manner to encoder updates.
   
   **Note (Work in Progress)**: We're aware that **data capping is the main limitation** of our current approach. As pointed out by Professor Basile, we're conducting comparative experiments to verify whether a larger model (e.g., `roberta-large`) fine-tuned on the entire dataset concatenation (without capping) could lead to superior performance. Preliminary tests are ongoing and will be discussed subsequently.
5. **The choice of simplicity: Hate Speech yes or no?**: Initially the idea was to add a fourth head for **Hate Speech**, thinking that "more tasks = smarter model." However, thinking it through better, we understood we'd only add noise. Hate Speech and Toxicity are concepts so close that the risk of creating ambiguity was very high. Instead of overloading the model with redundant tasks, we chose the path of minimalism: better a lighter model focused on the main objective (OLID and implicit toxicity detection in text) than one unnecessarily complicated by nearly identical signals.

**V8 Result**: F1 **0.77**.
The score was similar to "cheating" V4, but this time the Overfitting Gap had dropped from 18% to **3.2%**. The model was finally honest.

**From V8 to V9: Less is More**
With **V9** we cleaned up. We definitively abandoned the Hate Speech idea because we understood that, at that moment, it would only "confuse" the gradient mix without bringing real advantages. We preferred to concentrate all the model's power on three clear and distinct fronts: **Toxicity**, **Emotions** (which we expanded to 7 classes to be more precise) and **Sentiment**. It was the necessary step to understand that the right path wasn't adding tasks randomly, but removing ambiguities to let the encoder breathe.

However, a **structural limitation** remained: all tasks passed through the same BERT encoder without specialization. Patterns useful for one task (e.g., irony in Sentiment) could "pollute" representations of another (e.g., direct toxicity). To overcome this bottleneck, we designed V10's **Task-Specific Attention** architecture.

---

### 3.4 Phase 4: The Architectural Breakthrough (V10)
We reached the conclusion that no loss function, however intelligent, could solve a problem at the foundation: the model's structure. The problem wasn't how to balance tasks, but the fact that tasks were "looking" at each other in a confused way.

We decided to completely change course, seeking a way to **separate representations**.

#### 3.4.1 Backbone Change: Why RoBERTa?
We abandoned BERT-base to switch to **`twitter-roberta-base`**. This choice was driven by practical considerations related to our task's domain:
1. **Pre-training Data**: RoBERTa was trained on social data (154 million tweets). It understands slang, abbreviations and informal style much better than standard BERT (trained on Wikipedia and books).
2. **BPE Robustness**: RoBERTa's BPE (Byte-Pair Encoding) tokenizer doesn't give up on masked words. It can reconstruct the meaning of *"f**k"* or *"st*pid"* by analyzing bytes, where BERT only saw an unknown signal ([UNK]).

#### 3.4.2 The real quality leap: Task-Specific Multi-Head Attention (TS-MHA)
We decided to implement this solution by applying the **Multi-Head Attention** principle studied in **Module 2 (Prof. Basile)**, particularly the concept of **"Redundancy"**: instead of having a single shared attention mechanism, we created **4 parallel and specialized Multi-Head Attention blocks** (one for each task). Initially we were parsimonious (thinking RoBERTa already had enough internal "redundancy" with its 144 heads), but qualitative tests showed us that forcing everything into the [CLS] vector created too much confusion. With 4 separate blocks, we gave each task its own specialized "eyes."

This allowed us to look at the entire sentence and isolate what's needed for each task, finally obtaining a clean result.

We used the **"8 Investigators"** metaphor:
- Each attention block has 8 heads. In the Toxicity one, for example, one head can learn to look for the subject (*"You"*, *"Y'all"*), another for aggressive verbs, yet another for negations.
- These "investigators" talk to each other (Self-Attention) and create a text representation specific to that problem.

#### 3.4.3 The Role of Reporting: "POS Eventive Shades"
For the Reporting task, we went beyond simple quotation mark search. Inspired by Sprugnoli's theory, we tried to have the model learn the **"eventive shades"** of parts of speech.

**Synthetic Dataset and Pseudo-Subjectivity**: Since no public dataset exists for Reporting Detection, we generated a **synthetic dataset** using **3 different Large Language Models** (GPT-OSS 120B, Claude 4.5 (thinking), Gemini 3 Pro (High)). The goal was to simulate **perspectivism** even in synthetic data: each model generated variants of toxic phrases (direct) and their "reported" versions (quoted). By combining annotations from the 3 LLMs, we obtained **pseudo-subjectivity** that reflects different "visions" of the reporting concept, reducing the bias of a single generative model. The final dataset contains ~1000 unique samples after aggressive deduplication.

The model must distinguish between the event as *action* (*"He insulted me"*) and the event as *reported object* (*"The insult was reported"*). Our dedicated attention head (in theory) learns to recognize "reporting anchors," those verbs and nouns that signal a functional transition from a statement to a quotation.

#### 3.4.4 Hierarchical Optimization
We resolved the war between different task gradients with a two-level strategy, applying **Advanced Learning** concepts from **Module 1**:
1. **Micro Level**: We use **Focal Loss** ($\gamma=2.0$) for Toxicity and Sentiment to "silence" easy examples and force the model to look only at **Hard Negatives** (aggressive but non-toxic phrases).
2. **Macro Level**: We use **Kendall Uncertainty** to balance tasks among themselves. In V10.2 we replaced the standard exponential function with **Softplus**, making the model much more stable and immune to gradient explosions during gradual unfreezing of RoBERTa weights.

**Progressive Unfreezing (Head Stabilization)**:
A fundamental technical choice to avoid **Catastrophic Forgetting** was implementing **progressive encoder freezing**:
- **Epoch 1**: RoBERTa is completely **frozen** (`requires_grad=False` for all encoder parameters). Only the 4 Task-Specific MHA heads and classification heads are trained. This allows heads to "learn their task" without ruining pre-trained weights.
- **Epochs 2-15**: RoBERTa is **unfrozen** (`requires_grad=True`), but with a very conservative learning rate (1e-5 vs 5e-5 for heads). The encoder can now finely adapt to our tasks without forgetting the linguistic knowledge learned on 154M tweets.

This strategy was crucial: in preliminary tests without initial freezing, the encoder "collapsed" in the first epochs because chaotic gradients from random heads destabilized it. With Epoch 1 freezing, heads align first, and then the encoder can adapt in a controlled manner.

Additionally, we implemented **Differential Learning Rates** and **Learning Rate Scheduling (Linear Warmup)** precisely to manage the different learning speeds between the pretrained backbone (a "Sage" requiring low LRs) and random heads (the "Apprentices" who must learn quickly), avoiding the *Catastrophic Forgetting* risk discussed in class.

---

### 3.5 Phase 5: Final Touches (V10.2 - January 28, 2026)

Everything seemed ready. Training was running well and results were excellent. However, that **F1 of 1.0** on the "Reporting" task kept bothering us: in statistics, if everything is perfect, there's usually a hidden error. We decided to check more carefully, and we found the last two issues.

#### 3.5.1 The data problem (Template Leakage)
We went through the "Reporting" dataset CSVs and started comparing phrases one by one. It was an unpleasant surprise to discover that our examples were full of near-duplicates.
For example:
- Train: *"You are stupid"*
- Validation: *"you are stupid"* (only capitalization changed)

The model wasn't learning reporting logic, it was just memorizing phrases. It was frustrating to have to delete half the data a few hours before submission, but we had no choice if we wanted serious work.

**Solution**: We wrote an aggressive "fuzzy" deduplication script. We normalized all text and removed any semantic overlap between train and validation. We dropped to **~1000 unique samples**, but finally had an honest test.

#### 3.5.2 The gradient bug (Phantom Gradients)
While fixing the data, we noticed that sometimes training would "go crazy". After considerable debugging, we found the culprit in Kendall Loss.

When a batch had no examples for a certain task (like Reporting), the model still tried to update the uncertainty $(\sigma)$ for that task. This created strange gradients that messed up the entire encoder.

**V10.2 Solution**: The final fix was introducing the **Task Mask**.
```python
# V10.2: The final blow to the bug
term = precision * loss + 0.5 * softplus(log_var)
total += term * mask[i]  # The mask blocks superfluous gradients
```
From that moment, training became a straight and stable line toward convergence.

#### 3.5.3 Final Training Configuration (V10.2)
For transparency, we report the exact configuration used for final training on **Kaggle T4 GPU**:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Batch Size** | 16 | VRAM limit (16GB). |
| **Gradient Accumulation** | 4 | Effective batch = 64, more stable without increasing memory. |
| **Epochs** | 15 | With Early Stopping (patience=5) triggered after 14 epochs. |
| **Learning Rate (Encoder)** | 1e-5 | Extremely conservative to preserve RoBERTa's pre-trained weights. |
| **Learning Rate (Task Heads)** | 5e-5 | 5x faster than encoder: heads are random and must converge quickly. |
| **Weight Decay** | 0.01 | Light L2 regularization. |
| **Focal Loss $\gamma$** | 2.0 | Standard for imbalanced datasets (Lin et al., 2017). |
| **Dropout** | 0.3 | Higher than RoBERTa default (0.1) for MTL, prevents overfitting on heads. |
| **Max Sequence Length** | 128 tokens | Sufficient for tweets and short posts. |

---

## 4. Final Results (V10.2)

| Task | Validation F1 | Class-Specific Detail (Epoch 9) |
|------|---------------|-------------------------------|
| **Toxicity** | **0.7572** | [Neg] F1: 0.81 \| [Pos] F1: 0.70 |
| Reporting | 1.00 | P: 1.00 \| R: 1.00 |
| Sentiment | Auxiliary | Not tracked (semantic support) |
| Emotion | Auxiliary | Not tracked (perspectivism support) |

### 4.2 Comparison with Previous Versions

| Version | F1 Score | Overfitting Gap | Verdict |
|---------|----------|-----------------|---------|
| V3 (Baseline) | 0.72 | ~5% | Functional but blind |
| V4 (Naive Kendall) | **0.78** | **18.4%** | **Discarded** (memorization) |
| V8 (Data Balanced) | 0.77 | 3.2% | Stable but weak architecture |
| **V10.2 (Final)** | **0.7572** | **~2%** | **Gold Standard** |

**Important note**: Our final F1 (0.7572) is slightly lower than V4's "peak" (0.78). This is a **conscious choice**. We prefer a model that generalizes honestly over one that cheats through overfitting.

### 4.3 Technical Deep Dive: Training Mathematical Analysis

Examination of V10.2 reveals fundamental technical patterns that justify our architectural choices:

#### 1. Class Balance (Toxicity Breakdown)
Unlike "biased" models, V10.2 shows remarkable ability to capture the positive class (Toxic):
- **Toxicity [Pos] (Epoch 9)**: Precision 0.63 | **Recall 0.79**.
- The fact that Recall (0.79) is higher than Precision (0.63) indicates we configured the model to be a "prudent" filter: we prefer some false positives (non-toxic phrase flagged) rather than letting a real insult pass (false negative).

#### 2. Evolution of Kendall Weights ($\sigma$)
Monitoring learnable homoscedastic uncertainty parameters was fundamental. Here's how the weights ($ \log \sigma^2 $) changed between Epoch 1 and Epoch 9:

| Task | Initial Weight (Ep. 1) | Final Weight (Ep. 9) | Trend and Interpretation |
|------|------------------------|---------------------|--------------------------|
| **Toxicity** | **1.435** | **1.305** | **Decreasing**: Model reduced uncertainty, increasing gradient importance. |
| **Emotion** | 1.431 | 1.300 | **Decreasing**: Same trend, emotional signal became a solid foundation. |
| **Sentiment** | 1.460 | 1.760 | **Increasing**: Increased uncertainty. Sentiment was too noisy compared to primary task. |
| **Reporting** | 1.437 | 1.591 | **Increasing**: Model understood the task was "solved" and reduced its impact. |

#### 3. Early Stopping Rationale
Training stopped at **Epoch 14** (Trigger: Patience 5/5) because the best F1 was reached at **Epoch 9**. Between epoch 9 and 14, we observed **Toxicity Loss still decreasing** (from 0.39 to 0.29) while validation F1 remained flat. This is the classic signal of beginning memorization (overfitting): the model was "perfecting" the loss on seen data without gaining more abstraction capability. Stopping at epoch 9 was the correct choice to preserve generalization.

---

## 5. Error Analysis

### 5.1 Where the Model Works Well (Victory over V1)

- **Direct insults**: *"Sei un idiota"* (*"You are an idiot"*) → Correctly flagged.
- **Negative non-toxic sentiment**: *"Odio il lunedì"* (*"I hate Mondays"*) → Correctly ignored.
- **Triumph over negations**: *"Non penso che tu sia stupido"* (*"I don't think you are stupid"*) → V10, thanks to 8 attention heads, "sees" the link between *not* and *stupid* and classifies it as **Non-Toxic**, where the baseline collapsed.
- **Quotations**: *"Ha detto 'sei un idiota'"* (*"He said 'you are an idiot'"*) → The Reporting head recognizes the framing and neutralizes the insult.

### 5.2 Where the Model Still Fails (The Final Frontier)

- **Implicit hate and dehumanization**: *"Gente come te dovrebbe stare allo zoo"* (*"People like you belong in a zoo"*) or *"People like you belong in a cage"*. Here the model fails. There are no "toxic" words in the standard dictionary, and dehumanization is a semantic concept too deep for RoBERTa without an external knowledge base.
- **Subtle sarcasm**: *"Complimenti, sei proprio un genio!"* (said with offensive intent). Without tone of voice or conversation context, the model only sees a compliment and positive sentiment.

These limitations confirm that current technology, however advanced, still struggles with **pragmatic inference**. It would be interesting, in the future, to integrate *Reasoning* models (like those based on knowledge graphs) to capture these edge cases.

---

## 6. Final Reflections

### What we're taking away
1. **Don't trust numbers too much**. Aiming for a very high F1 without looking at overfitting is useless. We learned that an "honest" 0.75 is worth much more than a 0.80 obtained by cheating or memorizing data.

2. **Architecture matters more than parameters**. We spent days looking for the magic formula for loss, when the real breakthrough was giving the model "different eyes" with Task-Specific Attention. If the model's structure is wrong, no parameter will help.

3. **Data is the hardest part**. That "1.0" on Reporting was a great lesson. Synthetic data can easily deceive if you don't do deep and manual cleaning.

4. **Context is everything**. Understanding if a phrase is toxic or just a quotation isn't a matter of keywords, but how those words are put together.

### What We Would Do Differently

If we could go back to the first day of lab:
- **Structure Immediately**: We'd implement Task-Specific Attention as the first move, instead of considering it an advanced option.
- **Data-First**: We'd spend much more time cleaning and analyzing datasets before launching hours of training on Kaggle.
- **Simplicity**: We wouldn't seek hyper-complex solutions (like the original Kendall) without first understanding baseline limitations.
- **Dataset Union**: As a final reflection, also following a suggestion from Professor Basile, probably instead of limiting ourselves to sampling auxiliary data (GoEmotions, SST-2) to protect OLID, we could have experimented with total dataset concatenation on an even more capable encoder. This could have unlocked even better performance by leveraging every single available sample.

---

## 7. Attachments

1. **AURA_V10_KAGGLE.ipynb** — Complete notebook (Kaggle-ready)
2. **aura-v10-data/** — Clean and deduplicated dataset
3. **AURA_README.md** — Complete technical documentation

---

*Report compiled on January 28, 2026, at 23:59.*
*Thank you for reading this far.*
