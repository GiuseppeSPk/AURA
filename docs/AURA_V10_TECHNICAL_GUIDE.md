# AURA V10: Technical Documentation
## System Architecture & Implementation Guide

---

# 1. Architettura Generale

## 1.1 Overview

AURA V10 è un sistema di **Multi-Task Learning** per l'analisi del testo che esegue 4 task simultaneamente:

| Task | Tipo | Output | Descrizione |
|:-----|:-----|:-------|:------------|
| Toxicity | Binary | [Non-Toxic, Toxic] | Rileva contenuto offensivo |
| Emotion | Multi-label | 7 emozioni Ekman | anger, disgust, fear, joy, sadness, surprise, neutral |
| Sentiment | Binary | [Negative, Positive] | Polarità generale |
| Reporting | Binary | [Direct, Reporting] | Distingue citazione da asserzione |

## 1.2 Schema Architetturale

```
                         INPUT: "He said you are an idiot"
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │     RoBERTa-base Encoder        │
                    │     (125M parameters)           │
                    │     Pre-trained on 160GB text   │
                    └─────────────────┬───────────────┘
                                      │
                              [768-dim vectors]
                              (per ogni token)
                                      │
        ┌─────────────┬───────────────┼───────────────┬─────────────┐
        │             │               │               │             │
        ▼             ▼               ▼               ▼             
   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ TOX-MHA │   │ EMO-MHA │   │SENT-MHA │   │ REP-MHA │
   │(8 heads)│   │(8 heads)│   │(8 heads)│   │(8 heads)│
   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
        │             │               │               │
   [Mean Pool]   [Mean Pool]   [Mean Pool]   [Mean Pool]
        │             │               │               │
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │ Linear  │   │ Linear  │   │ Linear  │   │ Linear  │
   │ 768→2   │   │ 768→7   │   │ 768→2   │   │ 768→1   │
   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
        │             │               │               │
        ▼             ▼               ▼               ▼
   [Toxic/Not]  [7 Emotions]   [Pos/Neg]    [Report?]
```

---

# 2. Task-Specific Multi-Head Attention (TS-MHA)

## 2.1 Teoria

### Il Problema del Shared Representation
Nel Multi-Task Learning tradizionale, tutti i task condividono le stesse rappresentazioni. Questo causa **interferenza negativa**:
- Un task che cerca pattern lessicali (toxicity) compete con uno che cerca pattern sintattici (reporting)

### La Soluzione: Feature Disentanglement
AURA V10 introduce **blocchi di attenzione separati** per ogni task. Ogni blocco impara a "guardare" parti diverse dell'input.

## 2.2 Codice Annotato

```python
class TaskSpecificMHA(nn.Module):
    """
    Multi-Head Self-Attention per singolo task.
    Implementa il 'Redundancy Principle' del Module 2.
    """
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        
        # === MULTI-HEAD ATTENTION ===
        # Formula: Attention(Q, K, V) = softmax(QK^T / √d_k) · V
        # - Q (Query): "Cosa sto cercando?"
        # - K (Key): "Cosa contiene ogni token?"
        # - V (Value): "Che informazione porto?"
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,   # 768 (dimensione RoBERTa)
            num_heads=n_heads,      # 8 teste parallele
            batch_first=True,       # Shape: (batch, seq, dim)
            dropout=dropout
        )
        
        # === LAYER NORMALIZATION ===
        # Normalizza le attivazioni per stabilità
        # Formula: LayerNorm(x) = γ · (x - μ) / σ + β
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask):
        # Maschera per ignorare token di padding
        key_padding_mask = (attention_mask == 0)
        
        # Self-Attention: Q = K = V = hidden_states
        # Il modello "interroga se stesso" per trovare relazioni
        attn_output, attn_weights = self.mha(
            query=hidden_states, 
            key=hidden_states, 
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        
        # === RESIDUAL CONNECTION + NORMALIZATION ===
        # Pattern Transformer: x' = LayerNorm(x + Dropout(Attention(x)))
        # Previene vanishing gradient in reti profonde
        output = self.layernorm(hidden_states + self.dropout(attn_output))
        
        return output, attn_weights
```

## 2.3 Le 8 Teste Spiegate

Ogni blocco MHA ha **8 teste parallele**. Ogni testa lavora su una porzione della dimensione:

```
768 dimensioni totali ÷ 8 teste = 96 dimensioni per testa
```

**Esempio pratico** per la frase *"He said you are an idiot"*:

| Testa | Pattern cercato | Tokens con alto attention |
|:-----:|:----------------|:--------------------------|
| 1 | Soggetto-Verbo | "He" ↔ "said" |
| 2 | Verbo-Citazione | "said" → "you are an idiot" |
| 3 | Pronome-Target | "you" → bersaglio insulto |
| 4 | Insulti lessicali | "idiot" |
| 5 | Intensificatori | (nessuno qui) |
| 6 | Negazioni | (nessuna) |
| 7 | Connettori | (nessuno) |
| 8 | Pattern posizionali | Inizio vs fine frase |

Le 8 prospettive vengono **concatenate e proiettate** per ottenere l'output finale.

## 2.4 Perché 4 Blocchi Separati?

Nel modello finale, i 4 blocchi TS-MHA sono **indipendenti**:

```python
# Ogni task ha il suo "investigatore" specializzato
self.tox_mha = TaskSpecificMHA(...)   # Cerca insulti diretti
self.emo_mha = TaskSpecificMHA(...)   # Cerca lessico emotivo
self.sent_mha = TaskSpecificMHA(...)  # Cerca polarità
self.report_mha = TaskSpecificMHA(...)# Cerca verbi di citazione
```

**Vantaggio**: Il blocco `report_mha` può imparare a dare peso a "said" senza interferire con `tox_mha` che deve ignorarlo.

---

# 3. Multi-Task Learning con Kendall Uncertainty

## 3.1 Il Problema del Bilanciamento

Con 4 task, come bilanciare le loss?

| Approccio | Problema |
|:----------|:---------|
| Somma semplice | Task con loss più alta domina |
| Pesi fissi | Richiede tuning manuale |
| **Kendall (2018)** | Pesi appresi automaticamente ✓ |

## 3.2 La Formula

$$L_{total} = \sum_{i=1}^{4} \frac{1}{2\sigma_i^2} L_i + \log(\sigma_i)$$

Dove:
- $L_i$ = Loss del task i
- $\sigma_i$ = Incertezza del task i (**parametro appreso**)
- Il termine $\log(\sigma_i)$ è un **regolarizzatore**

**Intuizione**:
- Task con loss alta → Il modello aumenta $\sigma_i$ → Peso diminuisce
- Il regolarizzatore previene $\sigma \to \infty$
- **Equilibrio automatico** tra i task

## 3.3 Implementazione

```python
class UncertaintyLoss(nn.Module):
    """Kendall et al. (2018) - Homoscedastic Uncertainty."""
    
    def __init__(self, n_tasks=4):
        super().__init__()
        # log_vars = log(σ²) - parametri APPRESI via backprop
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, losses, mask=None):
        """
        Args:
            losses: Lista di 4 loss [L_tox, L_emo, L_sent, L_rep]
            mask: [1,1,1,1] se tutti i task presenti, [1,0,0,1] se solo tox e rep
        """
        total = 0
        if mask is None:
            mask = [1.0] * len(losses)
            
        for i, loss in enumerate(losses):
            # === PRECISION = 1/(2σ²) con SoftPlus per stabilità ===
            # SoftPlus(x) = log(1 + e^x) - sempre positivo
            precision = 1.0 / (F.softplus(self.log_vars[i]) + 1e-8)
            
            # === FORMULA KENDALL ===
            # precision * loss + log(σ) 
            term = precision * loss + F.softplus(self.log_vars[i]) * 0.5
            
            # === MASKING (Fix V10.2) ===
            # Se un task è assente nel batch, non contribuisce
            total += term * mask[i]
            
        return total
    
    def get_weights(self):
        """Visualizza i pesi correnti: 1/σ²"""
        return (1.0 / (F.softplus(self.log_vars) + 1e-8)).detach().cpu().numpy()
```

## 3.4 Il Fix V10.2: Phantom Gradients

**Problema originale**: In batch misti, alcuni task sono assenti (es. batch con solo samples di Emotion). La loss dummy `0.0` aveva `requires_grad=True`, causando aggiornamenti spuri ai log_vars.

**Soluzione**: Maschera che azzera l'intero termine (incluso il regolarizzatore):
```python
total += term * mask[i]  # Se mask[i]=0, niente contribuisce
```

---

# 4. Focal Loss

## 4.1 Problema: Class Imbalance

Nel dataset Toxicity: ~95% Non-Toxic, ~5% Toxic.
La Cross-Entropy standard "impara" a predire sempre la classe maggioritaria.

## 4.2 Soluzione: Focus sugli Errori Difficili

$$FL(p_t) = -(1 - p_t)^\gamma \cdot \log(p_t)$$

Dove:
- $p_t$ = Probabilità predetta per la classe corretta
- $\gamma$ = 2.0 (parametro di focusing)

**Comportamento**:
- Se $p_t = 0.9$ (facile): $(1-0.9)^2 = 0.01$ → Loss quasi azzerata
- Se $p_t = 0.1$ (difficile): $(1-0.1)^2 = 0.81$ → Loss alta

## 4.3 Codice

```python
def focal_loss(logits, targets, gamma=2.0, weight=None, smoothing=0.0):
    """
    Focal Loss: Focalizza l'apprendimento sugli esempi difficili.
    """
    # Cross-Entropy standard
    ce = F.cross_entropy(logits, targets, weight=weight, 
                         reduction='none', label_smoothing=smoothing)
    
    # p_t = probabilità della classe corretta
    pt = torch.exp(-ce)
    
    # Modulazione: (1-pt)^gamma riduce loss per esempi facili
    return ((1 - pt) ** gamma * ce).mean()
```

---

# 5. Training Loop

## 5.1 Gradient Accumulation

**Problema**: GPU memory limitata → batch_size piccolo → gradienti rumorosi

**Soluzione**: Accumulare gradienti su più step prima di aggiornare

```python
CONFIG = {
    'batch_size': 16,              # Batch fisico
    'gradient_accumulation': 4,    # Accumula 4 batch
    # Effective batch = 16 × 4 = 64
}
```

**Implementazione**:
```python
for step, batch in enumerate(train_loader):
    loss = compute_loss(batch) / CONFIG['gradient_accumulation']
    loss.backward()  # Accumula gradienti
    
    if (step + 1) % CONFIG['gradient_accumulation'] == 0:
        optimizer.step()   # Aggiorna pesi
        optimizer.zero_grad()  # Reset gradienti
```

## 5.2 Backbone Freezing

**Strategia**: Primo epoch con RoBERTa congelato

```python
if epoch <= CONFIG['freezing_epochs']:  # Epoch 1
    for p in model.roberta.parameters(): 
        p.requires_grad = False  # Congela backbone
else:
    for p in model.roberta.parameters(): 
        p.requires_grad = True   # Scongela
```

**Motivazione**:
- Le classification heads sono inizializzate random
- Se scongeliamo subito il backbone, gradienti rumorosi rovinano il pre-training
- Freezing permette alle heads di "stabilizzarsi" prima

## 5.3 Differential Learning Rates

```python
optimizer = torch.optim.AdamW([
    {'params': model.roberta.parameters(), 'lr': 1e-5},    # Backbone: LR basso
    {'params': heads_params, 'lr': 5e-5}                   # Heads: LR alto
], weight_decay=0.01)
```

**Motivazione**: Il backbone contiene conoscenza pre-trained, va modificato poco. Le heads partono da zero e devono imparare velocemente.

## 5.4 Early Stopping

```python
if val_f1 > best_f1:
    best_f1 = val_f1
    patience_counter = 0
    torch.save(model.state_dict(), 'best_model.pt')
else:
    patience_counter += 1
    if patience_counter >= CONFIG['patience']:  # 5 epochs
        print('Early stopping!')
        break
```

---

# 6. Mean Pooling vs [CLS]

## 6.1 Due Strategie

| Strategia | Formula | Pro | Contro |
|:----------|:--------|:----|:-------|
| [CLS] token | $h_{CLS}$ | Standard, veloce | Perde info sparse |
| Mean Pooling | $\frac{1}{N}\sum_i h_i$ | Cattura tutto | Diluisce focus |

## 6.2 Implementazione Mean Pooling

```python
def _mean_pool(self, seq, mask):
    """Media pesata dei token, escludendo padding."""
    # Espandi maschera: (batch, seq) → (batch, seq, dim)
    mask_exp = mask.unsqueeze(-1).expand(seq.size()).float()
    
    # Somma token validi / conteggio token validi
    return (seq * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
```

**Scelta per AURA**: Mean Pooling. Per testi brevi (tweets, commenti), cattura meglio il contesto distribuito.

---

# 7. Bias Initialization

## 7.1 Problema

Toxicity: 5% positivi. A inizio training, il modello predice 50/50 → Loss alta sulla classe maggioritaria → Converge verso "sempre Non-Toxic".

## 7.2 Soluzione

```python
with torch.no_grad():
    self.toxicity_head.bias[0] = 0.5    # Non-Toxic: bias positivo
    self.toxicity_head.bias[1] = -0.5   # Toxic: bias negativo
```

Questo inizializza le probabilità a ~62% Non-Toxic / ~38% Toxic, più vicino alla distribuzione reale.

---

# 8. Riepilogo Parametri

| Parametro | Valore | Motivazione |
|:----------|:------:|:------------|
| Encoder | roberta-base | Balance potenza/efficienza |
| Hidden dim | 768 | Standard RoBERTa |
| Attention heads | 8 | 768/8 = 96 dim/head |
| Dropout | 0.3 | Anti-overfitting aggressivo |
| Batch size | 16 | Limite GPU |
| Grad accumulation | 4 | Effective batch = 64 |
| LR encoder | 1e-5 | Preserva pre-training |
| LR heads | 5e-5 | Apprendimento veloce |
| Focal gamma | 2.0 | Standard Lin et al. |
| Label smoothing | 0.1 | Generalizzazione |
| Patience | 5 | Early stopping |
| Freeze epochs | 1 | Stabilizza heads |

---

# 9. Risultati Attesi

## Metriche Target

| Task | Metrica | Target | Note |
|:-----|:--------|:------:|:-----|
| Toxicity | F1 Macro | >0.75 | Bilanciata tra classi |
| Emotion | F1 Sample | >0.50 | Multi-label difficile |
| Sentiment | F1 Macro | >0.80 | Task più semplice |
| Reporting | F1 Macro | >0.65 | Dataset limitato |

## Kendall Weights (Fine Training)

I pesi appresi indicano la "fiducia" del modello per task:
- Toxicity: ~1.0 (task principale, alta fiducia)
- Emotion: ~0.5 (multi-label, incerto)
- Sentiment: ~1.2 (task facile)
- Reporting: ~0.7 (dataset piccolo)

---

*Documento generato per supporto alla discussione tesi.*
