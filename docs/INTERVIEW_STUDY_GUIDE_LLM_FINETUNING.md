# 📚 Guida Completa al Colloquio: LLM Fine-Tuning @ LINKS

> **Candidato**: Giuseppe Spicchiarello  
> **Posizione**: Internship su LLM Fine-Tuning (Modelli <3B parametri)  
> **Focus**: Catastrophic Forgetting & Knowledge Acquisition vs Retention  
> **Data Colloquio**: 9 Febbraio 2026

---

# PARTE 1: CONCETTI FONDAMENTALI

## 1.1 Cos'è il Fine-Tuning di un LLM?

**Definizione**: Processo di adattamento di un modello pre-addestrato (come BERT, RoBERTa, LLaMA) a un task specifico usando un dataset più ristretto e domain-specific.

**Formula concettuale**:
```
Modello Fine-Tuned = Pretrained Weights + Task-Specific Adaptation
```

**Differenza chiave**:
| | Training from Scratch | Fine-Tuning |
|---|---|---|
| **Dati richiesti** | Milioni/miliardi | Migliaia |
| **Tempo** | Settimane/mesi | Ore/giorni |
| **Learning Rate** | Standard (1e-3) | Molto basso (1e-5 / 1e-6) |
| **Rischio** | Underfitting | Catastrophic Forgetting |

---

## 1.2 Catastrophic Forgetting

**Definizione**: Fenomeno per cui un modello, durante il fine-tuning, "dimentica" le conoscenze acquisite durante il pre-training.

**Causa**: I gradienti del fine-tuning sovrascrivono i pesi che codificavano la conoscenza generale.

**Esempio pratico**:
- Un LLM sa rispondere a domande generali
- Dopo fine-tuning su dati medici, risponde bene a domande mediche
- Ma "dimentica" come rispondere a domande generali

### Tecniche di Mitigazione

| Tecnica | Descrizione | Quando usarla |
|---------|-------------|---------------|
| **Layer Freezing** | Congelare i primi layer dell'encoder | Sempre, per modelli piccoli |
| **Low Learning Rate** | LR 1e-5 o 1e-6 per il backbone | Standard per fine-tuning |
| **Elastic Weight Consolidation (EWC)** | Penalizza modifiche a pesi "importanti" | Continual learning |
| **Replay-based Methods** | Mescolare dati originali con nuovi dati | Quando si ha accesso ai dati originali |
| **LoRA/QLoRA** | Adattatori a basso rango | Modelli molto grandi (>7B) |
| **Progressive Unfreezing** | Sbloccare gradualmente i layer | Approccio bilanciato |

---

## 1.3 Trade-off: Knowledge Acquisition vs Retention

**Il dilemma centrale**:
- **Più imparo** (acquisition) → **Più dimentico** (forgetting)
- **Meno imparo** → **Mantengo la conoscenza originale** ma non mi adatto

### Strategie di Bilanciamento

1. **Differential Learning Rates**:
   ```python
   optimizer = AdamW([
       {'params': backbone.parameters(), 'lr': 1e-5},  # Preserva
       {'params': heads.parameters(), 'lr': 5e-5}     # Apprende
   ])
   ```

2. **Gradual Unfreezing** (dal progetto AURA):
   - Epoca 1: Backbone congelato, solo heads addestrate
   - Epoche 2+: Backbone sbloccato con LR basso

3. **Monitorare le metriche su benchmark standard**:
   - Valutare su task originali (GLUE, MMLU) durante il fine-tuning
   - Se calano troppo → troppo forgetting

---

# PARTE 2: TECNICHE AVANZATE (DA AURA)

## 2.1 Multi-Task Learning

**Approccio AURA**: Addestrare il modello su 4 task simultaneamente:
- Toxicity Detection (primario)
- Emotion Classification
- Sentiment Analysis
- Reporting Detection

**Vantaggio**: Le rappresentazioni condivise sono più robuste perché devono funzionare per tutti i task.

**Il problema del Task Interference**:
Ogni task "compete" per i gradienti. Soluzione: **Kendall Uncertainty Loss**.

---

## 2.2 Kendall Uncertainty Loss

**Paper**: Kendall et al., 2018 - *Multi-Task Learning Using Uncertainty to Weigh Losses*

**Formula**:
$$L_{total} = \sum_{i=1}^{n} \frac{1}{2\sigma_i^2} L_i + \log(\sigma_i)$$

**Intuizione**:
- $\sigma_i$ = incertezza per il task i (parametro APPRESO)
- Task con loss alta → $\sigma$ aumenta → peso diminuisce
- Il modello bilancia automaticamente i task

**Implementazione chiave (AURA V10.2)**:
```python
# Fix: Usare SoftPlus invece di exp() per stabilità
precision = 1.0 / (F.softplus(log_vars[i]) + 1e-8)
term = precision * loss + F.softplus(log_vars[i]) * 0.5
```

---

## 2.3 Focal Loss per Dataset Sbilanciati

**Paper**: Lin et al., 2017 - *Focal Loss for Dense Object Detection*

**Formula**:
$$FL(p_t) = -(1 - p_t)^\gamma \cdot \log(p_t)$$

Dove:
- $\gamma = 2.0$ (standard)
- $p_t$ = probabilità predetta per la classe corretta

**Comportamento**:
- Esempio facile ($p_t = 0.9$): $(1-0.9)^2 = 0.01$ → Loss quasi zero
- Esempio difficile ($p_t = 0.1$): $(1-0.1)^2 = 0.81$ → Loss alta

**Uso**: Classifica sbilanciata (5% Toxic vs 95% Non-Toxic)

---

## 2.4 Task-Specific Multi-Head Attention

**Problema**: In MTL classico, tutti i task condividono le stesse rappresentazioni → interferenza

**Soluzione AURA**: Blocchi di attenzione separati per ogni task

```
RoBERTa Encoder → [768-dim per token]
                        ↓
    ┌──────────┬────────┴────────┬──────────┐
    ▼          ▼                 ▼          ▼
[TOX-MHA]  [EMO-MHA]      [SENT-MHA]  [REP-MHA]
 8 heads    8 heads         8 heads    8 heads
    ↓          ↓                 ↓          ↓
[Toxic?]  [Emotion]       [Sentiment] [Report?]
```

**Vantaggio**: Ogni task ha i propri "occhi" specializzati sul testo.

---

# PARTE 3: MODELLI PICCOLI (<3B) CONSIGLIATI

| Modello | Parametri | Pro | Contro |
|---------|-----------|-----|--------|
| **Phi-2** | 2.7B | Microsoft, eccellente per reasoning | API limitata |
| **Gemma-2B** | 2B | Google, open-source, efficiente | Nuovo, meno documentato |
| **LLaMA-2-7B** | 7B | Meta, community enorme | Sopra threshold |
| **TinyLlama** | 1.1B | Ultra-leggero, facile da fine-tunare | Meno capace |
| **Mistral-7B** | 7B | Stato dell'arte, sliding window attention | Sopra threshold |
| **Qwen-1.8B** | 1.8B | Alibaba, buono per multilingua | Meno testato in EU |

**Raccomandazione per il tirocinio**:
- **Phi-2** o **Gemma-2B** per esperimenti rapidi
- **TinyLlama** se serve ultra-efficienza

---

# PARTE 4: PARAMETER-EFFICIENT FINE-TUNING (PEFT)

## 4.1 LoRA (Low-Rank Adaptation)

**Paper**: Hu et al., 2021

**Idea**: Invece di aggiornare tutti i pesi W, aggiungere matrici a basso rango:
$$W' = W + BA$$

Dove:
- $W$ = pesi originali (congelati)
- $B, A$ = matrici di rango $r$ (trainabili)
- $r \ll d$ (tipicamente 4-16)

**Vantaggio**: Riduce parametri trainabili del 99%+

## 4.2 QLoRA

**Aggiunta**: Quantizzazione a 4-bit del modello base + LoRA

**Beneficio**: Fine-tuning di modelli 7B su GPU consumer (8GB VRAM)

---

# PARTE 5: VALUTAZIONE DEL FORGETTING

## 5.1 Metriche da Monitorare

| Metrica | Cosa Misura | Come Calcolarla |
|---------|-------------|-----------------|
| **Retention Score** | Performance su task originali | F1 su benchmark pre-FT / F1 post-FT |
| **Acquisition Score** | Performance su nuovo task | F1 su validation set domain-specific |
| **Overfitting Gap** | Differenza train/val | (Train Acc - Val Acc) / Train Acc |

## 5.2 Benchmark Standard

- **GLUE/SuperGLUE**: Comprensione del linguaggio generale
- **MMLU**: Multi-task multiple choice (conoscenza generale)
- **HellaSwag**: Commonsense reasoning
- **TruthfulQA**: Verità fattuale

**Approccio consigliato**:
1. Valutare modello pre-fine-tuning su MMLU/GLUE
2. Fine-tune su dati dominio
3. Ri-valutare su MMLU/GLUE
4. Calcolare retention = post_score / pre_score

---

# PARTE 6: LA TUA ESPERIENZA RILEVANTE (DA CV)

## 6.1 Progetti Chiave

### 🧠 P.E.R.S.O.N.A. Framework
**Concept**: Rilevamento di "Alignment Faking" tramite analisi neuro-simbolica.
- **Divergenza rilevata**: I modelli (Llama 3.1) usano retorica deontologica in linguaggio naturale ma logica utilitaristica nel codice.
- **Hypocrisy Index (H)**: Misura questa divergenza usando semantic embeddings.
- **Collegamento Interview**: Questo dimostra capacità di analisi profonda sulla **divergenza di comportamento** post-training, un tema centrale quando si parla di "Retention" vs "Acquisition".

### 🛡️ VERITY Platform
**Concept**: Piattaforma di Red Teaming per Robustezza Adversariale e AI Control.
- **Tecnologie**: FastAPI, Docker, Multi-agent attacks (PAIR, TAP).
- **Pipeline**: LLM-as-a-Judge con intervalli di confidenza statistica (CI 95%).
- **Collegamento Interview**: Dimostra competenza nella **valutazione rigorosa** e nel test di sicurezza, fondamentale per verificare che il fine-tuning non abbia introdotto vulnerabilità o rimosso "safety guardrails".

## 6.2 Come Collegare al Tirocinio

| Concetto LINKS | Tua Esperienza (AURA / CV) |
|----------------|---------------------|
| "Minimizzare forgetting" | AURA: Progressive unfreezing, differential LR |
| "Scalable Oversight" | VERITY: Pipeline LLM-as-a-Judge automatizzata |
| "Alignment Faking" | PERSONA: Hypocrisy Index e analisi codice |
| "Dataset sbilanciati" | AURA: Focal Loss (γ=2.0) |
| "Robustezza" | VERITY: Attacchi PAIR/TAP e jailbreak detection |

## 6.3 Frasi Chiave Personalizzate

> *"Nel progetto **PERSONA** ho analizzato come il fine-tuning possa indurre 'Alignment Faking', dove il modello mantiene una retorica sicura ma implementa logica divergente nel codice. Ho quantificato questo fenomeno con un Hypocrisy Index basato su embeddings."*

> *"Con **VERITY**, ho sviluppato sistemi di Scalable Oversight usando pipeline 'LLM-as-a-Judge'. Questa esperienza è direttamente applicabile alla valutazione della retention di safety guardrails durante il fine-tuning di modelli piccoli."*

> *"In **AURA**, ho gestito il catastrophic forgetting in un sistema multi-task, portando l'overfitting gap dal 18% al 2% grazie a una strategia di progressive unfreezing e l'uso di Kendall Uncertainty Loss."*

---

# PARTE 7: DOMANDE PROBABILI E RISPOSTE

## Q1: "Cos'è il catastrophic forgetting?"
> **R**: È il fenomeno per cui un modello, durante il fine-tuning, sovrascrive i pesi che codificavano conoscenza generale. Si mitiga con learning rate bassi, layer freezing progressivo, o tecniche come EWC e replay methods.

## Q2: "Come bilanceresti acquisition e retention?"
> **R**: Con un approccio a due livelli:
> 1. **Learning rate differenziali**: backbone a 1e-5, nuovi layer a 5e-5
> 2. **Monitoraggio continuo**: valutare su benchmark standard (MMLU, GLUE) durante il training per rilevare forgetting precoce
> 3. **Early stopping basato su retention**: se la performance su benchmark generali cala oltre soglia, fermasi

## Q3: "Che esperienza hai con il fine-tuning?"
> **R**: Nel progetto AURA ho fatto fine-tuning di RoBERTa-base per 4 task simultanei. Ho implementato Kendall Uncertainty Loss per bilanciare i task e Focal Loss per classi sbilanciate. Il risultato è stato F1 0.757 con overfitting gap del 2%.

## Q4: "Quali modelli useresti per esperimenti su <3B parametri?"
> **R**: Per esperimenti rapidi, **Phi-2** (2.7B) o **Gemma-2B** sono ottime scelte. Se serve ultra-efficienza, **TinyLlama** (1.1B). Per questi modelli, consiglierei QLoRA per ridurre i parametri trainabili e l'uso di GPU.

## Q5: "Come misureresti il forgetting?"
> **R**: Comparando le performance su benchmark standard (MMLU, GLUE) prima e dopo il fine-tuning. Calcolo un "retention score" = F1_post / F1_pre. Se scende sotto 0.9, c'è forgetting significativo.

---

# PARTE 8: ANTI-OVERFITTING CHECKLIST

☐ **Learning Rate basso** per backbone (1e-5)  
☐ **Dropout aumentato** (0.3-0.4 per MTL)  
☐ **Weight Decay** (0.01)  
☐ **Early Stopping** con patience (5 epoche)  
☐ **Gradient Accumulation** per effective batch più grande  
☐ **Label Smoothing** (0.1)  
☐ **Data Augmentation** dove possibile  
☐ **Deduplicazione** train/val per evitare data leakage  

---

# PARTE 9: GLOSSARIO RAPIDO

| Termine | Definizione |
|---------|-------------|
| **PEFT** | Parameter-Efficient Fine-Tuning |
| **LoRA** | Low-Rank Adaptation |
| **MTL** | Multi-Task Learning |
| **EWC** | Elastic Weight Consolidation |
| **Catastrophic Forgetting** | Perdita di conoscenza durante fine-tuning |
| **Knowledge Base** | Repository strutturato di informazioni dominio-specifiche |
| **RAG** | Retrieval-Augmented Generation |
| **Homoscedastic Uncertainty** | Incertezza uniforme per tutti i dati (Kendall) |
| **Focal Loss** | Loss che penalizza esempi facili |
| **Progressive Unfreezing** | Sblocco graduale dei layer |

---

# PARTE 10: RISORSE PER REFRESH VELOCE

## Papers da Rivedere (15 min cad.)
1. **LoRA**: [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
2. **Kendall Uncertainty**: [arxiv.org/abs/1705.07115](https://arxiv.org/abs/1705.07115)
3. **Focal Loss**: [arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

## Concetti da Ripassare
- Architettura Transformer (Attention, Q/K/V)
- BERT vs RoBERTa (differenze di pre-training)
- Tokenization (BPE, WordPiece)
- Ottimizzatori (Adam, AdamW, differenze)

---

# PARTE 11: BENCHMARK SUGGERITI PER IL FUTURO

Se ti chiedono: *"Come valuteresti ulteriormente la robustezza di AURA?"*, cita questi due dataset:

### 1. Toxigen (Adversarial & Machine-Generated)
- **Perché**: È un dataset di 270k frasi generate da LLM che sono **implicitamente tossiche** (senza parolacce) e mirano a "fregare" i classificatori.
- **Valore per AURA**: Testare la capacità dell'architettura di andare oltre il lessico e capire l'intent aggressivo filtrato da macchine.
- **Connessione a PERSONA**: Ottimo per misurare se il modello è "ipocrita" anche quando riceve input generati da altre AI.

### 2. AbuseEval (Implicit vs Explicit)
- **Perché**: È un'estensione di OLID che classifica l'abuso come `EXPLICIT` o `IMPLICIT`.
- **Valore per AURA**: Permette di dimostrare numericamente che la **Task-Specific Attention** migliora la recall sull'abuso implicito rispetto alle baseline standard (BERT/[CLS]).

---

**Buona fortuna domani! 🍀**

*Documento generato l'8 Febbraio 2026*
