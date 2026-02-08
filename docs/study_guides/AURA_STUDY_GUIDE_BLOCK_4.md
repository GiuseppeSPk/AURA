# AURA MASTER STUDY GUIDE: Block 4 (Deep Dive Edition)
## Ottimizzazione e Loss: Focal Loss, Kendall e il Fix del Phantom Gradient

In questo blocco analizziamo la "cabina di regia" di AURA: come il modello impara dai suoi errori e come bilancia 4 task simultaneamente senza impazzire.

---

### 1. Focal Loss: Gestire lo Sbilanciamento delle Classi

In AURA, la Focal Loss è l'arma segreta contro i dataset "sporchi" (dove il 95% è Non-Tossico).

#### 1.1 La Derivazione Matematica
Partiamo dalla Cross-Entropy standard ($CE = -\log p_t$). La Focal Loss aggiunge un **fattore di modulazione**:
$$ FL(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t) $$

- **$p_t$**: Probabilità assegnata alla classe corretta.
- **$\gamma$ (Focusing Parameter)**: Di solito impostato a 2.0.
  - Se il modello è sicuro ($p_t = 0.9$), il fattore $(1-0.9)^2 = 0.01$ riduce la perdita di 100 volte.
  - Se il modello sbaglia ($p_t = 0.1$), il fattore $(1-0.1)^2 = 0.81$ mantiene la perdita quasi intatta.
- **$\alpha$ (Alpha)**: Bilancia l'importanza tra la classe positiva (Tossico) e negativa.

> [!IMPORTANT]
> **Il Risultato**: AURA spende il 90% delle sue energie per correggere quel 5% di commenti difficili, invece di cullarsi sugli allori dei commenti facili.

---

### 2. Kendall Uncertainty: Il Bilanciamento Automatico (MTL)

In un sistema con 4 task, non possiamo sommare le loss semplicemente. Kendall et al. (2018) propongono di usare l'**incertezza omoschedastica** (legata al task) per pesare le perdite.

#### 2.1 La Formula di AURA
$$ L_{total} = \sum_{i=1}^{4} \left( \frac{1}{2\sigma_i^2} L_i + \log \sigma_i \right) $$

- **$L_i$**: Loss del task $i$ (es. Focal Loss per Toxicity).
- **$\sigma_i$**: Incertezza appresa per il task $i$.
- **$\frac{1}{2\sigma_i^2}$**: Dette anche "Precisione". Se il task è incerto ($\sigma$ alta), il suo peso nella loss totale diminuisce.
- **$\log \sigma_i$**: Regolarizzatore. Impedisce al modello di impostare $\sigma = \infty$ per rendere la loss totale zero.

---

### 3. Il Phantom Gradient Proof (Giallo Tecnico)

Durante la versione 10.1 di AURA, abbiamo scoperto che il modello imparava parametri di incertezza assurdi per i task assenti nel batch. Perché?

#### 3.1 La Prova Matematica
Immagina un batch che contiene solo dati di Toxicity. La loss per il Sentiment è $L_{sent} = 0$.
La formula per quel task diventa:
$$ \text{Loss}_{sent} = \frac{1}{2\sigma_{sent}^2} \cdot 0 + \log \sigma_{sent} = \log \sigma_{sent} $$

Se calcoliamo il gradiente rispetto a $\sigma_{sent}$:
$$ \frac{\partial \text{Loss}_{total}}{\partial \sigma_{sent}} = \frac{1}{\sigma_{sent}} $$

Il gradiente non è zero! L'ottimizzatore vede un modo per ridurre la loss totale: **diminuire $\sigma_{sent}$ verso lo zero**. 
Il modello sta "imparando" a diventare sicurissimo di un compito che non ha nemmeno visto! Questo è il **Phantom Gradient**.

#### 3.2 La Soluzione (V10.2)
Abbiamo introdotto una maschera binaria $M \in \{0, 1\}$ che annulla **l'intero termine**:
$$ L_{total} = \sum M_i \left( \frac{1}{2\sigma_i^2} L_i + \log \sigma_i \right) $$
Ora, se il task manca ($M_i=0$), il gradiente è esattamente zero.

---

### 4. Scheduler e Stabilità del Training

AURA usa un **OneCycleLR** scheduler. 
- Inizia con un LR basso per non sconvolgere RoBERTa.
- Sale a metà training per esplorare nuove configurazioni delle teste TS-MHA.
- Scende alla fine per "raffinare" i pesi dell'incertezza $\sigma$.

---

### AURA BLOCK 4: Optimization & MTL
## Technical Reference Module

1. **"Cosa accadrebbe se rimuovessi il termine $\log \sigma$ dalla formula di Kendall?"**
   *Risp: Il segnale di training diventerebbe instabile. Il modello cercherebbe di minimizzare la perdita portando $\sigma$ all'infinito (azzerando il termine $L/\sigma^2$), 'evadendo' di fatto il compito di imparare.*

2. **"Perché il Phantom Gradient è particolarmente pericoloso nel Multi-Task Learning?"**
   *Risp: Perché distorce l'equilibrio tra i task. Se un task assente vede la sua incertezza crollare artificialmente, quando riapparirà in un batch successivo avrà un peso (precisione) enorme, causando sbalzi violenti nei gradienti dell'encoder e rovinando il pre-training di RoBERTa.*

3. **"In quali casi la Focal Loss potrebbe essere dannosa?"**
   *Risp: Se il dataset è estremamente rumoroso (molte etichette sbagliate). In quel caso la Focal Loss costringerebbe il modello a focalizzarsi ossessivamente sugli errori che in realtà sono etichette errate, portando a un overfitting sul rumore.*

---

*Ultimo blocco: Metriche di Valutazione e Strategie di Validazione.*
