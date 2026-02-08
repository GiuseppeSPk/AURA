# AURA MASTER STUDY GUIDE: Block 3 (Deep Dive Edition)
## L'Architettura Task-Specific: TS-MHA, Mean Pooling e Teste di Classificazione

In questo blocco analizziamo come AURA V10 trasforma le rappresentazioni generiche di RoBERTa in decisioni specifiche per i 4 task (Toxicity, Emotion, Sentiment, Reporting).

---

### 1. Task-Specific Multi-Head Attention (TS-MHA)

Il cuore dell'innovazione di AURA è l'aggiunta di blocchi di attenzione separati per ogni compito.

#### 1.1 Perché non usare solo l'ultimo layer di RoBERTa?
I modelli pre-trained sono generalisti. Ogni task richiede di "guardare" la frase con filtri diversi:
- **Toxicity**: Sensibile a termini abusivi e target dell'insulto.
- **Reporting**: Sensibile a strutture sintattiche (verba dicendi, virgolette).
- **Emotion**: Sensibile a intensificatori e aggettivi carichi.

#### 1.2 La Teoria del Disentanglement
In AURA, ogni blocco `TaskSpecificMHA` riceve gli stessi hidden states dal backbone ma possiede le proprie matrici $W_Q, W_K, W_V$.
- Durante il training, il blocco specializzato per il **Reporting** imparerà a dare pesi altissimi a parole come *"disse"* o *"sostenne"*, mentre il blocco della **Toxicity** imparerà a ignorarle per non confondere una citazione con un attacco diretto.

---

### 2. Il Mean Pooling: Dal Vettore Token al Vettore Frase

Dopo che il blocco TS-MHA ha raffinato i vettori, dobbiamo ridurli a un unico vettore per la classificazione.

#### 2.1 La Matematica del Mascheramento
Non possiamo fare una media semplice perché la sequenza contiene token di `Padding` (zeri artificiali per pareggiare la lunghezza). Se usassimo il padding, la media verrebbe "diluita".

**Algoritmo di AURA**:
1. **Input**: Hidden States $H \in \mathbb{R}^{B \times L \times 768}$ e Attention Mask $M \in \{0, 1\}^{B \times L}$.
2. **Espansione**: $M$ viene trasformata in $M_{exp} \in \mathbb{R}^{B \times L \times 768}$.
3. **Zereamento**: $H_{masked} = H \odot M_{exp}$ (Moltiplicazione elemento per elemento).
4. **Somma e Media**: 
   $$ \text{S} = \sum_{i=1}^{L} H_{masked, i} \quad \text{,} \quad \text{Count} = \sum_{i=1}^{L} M_{exp, i} $$
   $$ \text{MeanPool} = \frac{\text{S}}{\text{Count} + \epsilon} $$

> [!TIP]
> **Perché $\epsilon$ (1e-9)?** Serve ad evitare la divisione per zero nel caso (impossibile ma teoricamente previsto) di una frase completamente vuota.

---

### 3. Classification Heads: L'Ultimo Miglio

Una volta ottenuto il vettore da 768 dimensioni tramite il pooling, AURA lo passa ai layer lineari finali:

| Task | Struttura Head | Attivazione / Loss |
|:---|:---|:---|
| **Toxicity** | Linear(768 → 2) | Softmax + Focal Loss |
| **Sentiment** | Linear(768 → 2) | Softmax + CrossEntropy |
| **Reporting** | Linear(768 → 1) | Sigmoid + BCE Loss |
| **Emotion** | Linear(768 → 7) | Sigmoid + Focal Loss (per multi-label) |

#### 3.1 Dropout: La Strategia Anti-Overfitting
Prima di ogni testa di classificazione, applichiamo un **Dropout del 30-40%**. 
- Durante il training, il modello "perde" casualmente il 40% delle informazioni del vettore di pooling. 
- Questo lo costringe a non basare la sua decisione su un singolo neurone "magico", ma a distribuire la conoscenza su tutto il vettore, rendendolo più robusto su dati nuovi.

---

### 4. Il Pattern Post-LayerNorm (Stabilità)

In AURA seguiamo l'architettura classica di RoBERTa dove la **LayerNorm** viene applicata *dopo* l'attenzione e la somma del residuo (Post-LN).
- **Vantaggio**: Mantiene il range delle attivazioni costante in ogni blocco, permettendoci di usare Learning Rate più alti senza che il modello diverga.

---

### AURA BLOCK 3: Architecture Deep Dive
## Technical Reference Module

1. **"Perché il Mean Pooling è teoricamente superiore al [CLS] token per testi social?"**
   Il token [CLS] è un aggregatore appreso, ma può soffrire di bias se il dataset di pre-training era diverso dal nostro. Il Mean Pooling, invece, garantisce che ogni singola parola della frase contribuisca alla decisione finale, il che è essenziale per catturare l'emozione o la tossicità sparsa in un commento lungo.

2. **"Come influisce la Dropout sul processo di generalizzazione di AURA?"**

3. **"Se un compito richiede analisi sintattica (come il Reporting), il Pooling è ancora utile?"**
   *Risp: Sì, perché il blocco TS-MHA del Reporting ha già rielaborato i vettori token includendo l'informazione sintattica tramite l'attenzione. Il Pooling poi riassume questa 'consapevolezza sintattica' in un unico segnale per il classificatore.*

---

*Prossimo blocco: Loss Functions Avanzate (Focal e Kendall Uncertainty).*
