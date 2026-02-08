# AURA MASTER STUDY GUIDE: Block 5 (Deep Dive Edition)
## Metriche e Valutazione: Oltre l'Accuracy

In questo blocco finale, analizziamo come AURA V10 viene valutata. La scelta delle metriche non è estetica: è una decisione ingegneristica per garantire la sicurezza del modello.

---

### 1. La Decomposizione dell'Errore: Matrice di Confusione

Per ogni task di AURA (es. Toxicity), costruiamo una matrice di confusione:

- **True Positive (TP)**: Il modello ha detto "Tossico" ed era vero.
- **False Positive (FP)**: Il modello ha detto "Tossico" ma era gentile (Censura inutile - costo alto per l'utente).
- **False Negative (FN)**: Il modello ha detto "Gentile" ma era tossico (Rischio sicurezza - costo alto per la piattaforma).
- **True Negative (TN)**: Il modello ha detto "Gentile" ed era vero.

---

### 2. Formule e Scelte Strategiche

#### 2.1 F1-Macro (Per i task sbilanciati)
In AURA usiamo l'**F1-Macro** per Toxicity e Sentiment.
$$ \text{F1-Macro} = \frac{\text{F1}_{Classe 0} + \text{F1}_{Classe 1}}{2} $$
Contrariamente alla media semplice (Micro), la Macro tratta la classe "Tossica" con la stessa dignità della classe "Non-Tossica". Se il modello ignora la classe più piccola, l'F1-Macro crolla sensibilmente.

#### 2.2 F1-Samples (Per le Emozioni)
Il task Emotion è **Multi-label**. Un commento può essere {Joy, Surprise}.
- L'F1-Sample calcola l'F1 per ogni singola frase (osservando l'intersezione tra etichette predette e reali).
- **Perché?** Ci dice quanto, in media, "abbiamo centrato il bersaglio" per ogni utente.

---

### 3. Soglie di Decisione: Softmax vs Sigmoid

- **Softmax (Toxicity/Sentiment)**: Le probabilità delle due classi sommano a 1. Scegliamo la classe con probabilità > 0.5.
- **Sigmoid (Emotion/Reporting)**: Ogni etichetta è indipendente (probabilità tra 0 e 1).
  - *Fine-tuning*: In AURA possiamo regolare la soglia (es. 0.4 invece di 0.5) per essere più "sensibili" a certe emozioni rare.

---

### 4. Bias Initialization (Trick per la Stabilità)

Perché AURA non "impazzisce" nei primi secondi di training? Grazie alla **Inizializzazione del Bias**.

**Problema**: Se il dataset è 95% Non-Tossico e inizializzi i pesi a caso, il modello all'inizio sbaglierà il 50% delle volte. Questo causerà una loss enorme nei primi batch, rischiando di cancellare i pesi utili di RoBERTa.

**Soluzione AURA**:
Inizializziamo il bias dell'ultimo layer lineare in modo che la probabilità iniziale del modello rifletta la distribuzione del dataset:
$$ b = \log \left( \frac{\pi}{1 - \pi} \right) $$
Dove $\pi$ è la percentuale di classe positiva (es. 0.05). In questo modo, il modello parte già sapendo che la tossicità è rara, e la loss iniziale è molto più bassa e controllabile.

---

### 5. Early Stopping e Validazione

AURA non si ferma quando la loss è zero (sintomo di overfitting), ma quando l'**F1-Macro sul Validation Set** smette di scendere per 5 epoch (Patience = 5).
- Questo garantisce che il modello sia in grado di generalizzare su commenti che non ha mai visto durante il training.

---

### AURA BLOCK 5: Metrics & Evaluation
## Technical Reference Module

1. **"Perché preferisce l'F1-Macro all'Accuracy in un progetto come AURA?"**
   *Risp: In dataset sbilanciati, l'Accuracy può essere virtualmente perfetta mentre il modello sta ignorando completamente la classe che ci interessa (Toxicity). L'F1-Macro assegna lo stesso peso all'importanza di ogni classe, forzando il modello a essere competente su tutto il dominio.*

2. **"Come interpreta una Precision molto alta ma una Recall molto bassa?"**
   *Risp: Il modello è 'prudente'. Dice che un commento è tossico solo quando ne è certissimo, ma ne sta perdendo molti per strada. È una scelta conservativa che evita la censura ingiusta ma riduce la sicurezza della piattaforma.*

3. **"Qual è l'impatto reale dell'inizializzazione del bias sul gradiente?"**
   *Risp: Riduce la magnitudo del gradiente nei primi step. Senza questo trick, l'errore enorme spingerebbe l'ottimizzatore a fare aggiornamenti violenti (gradiente esplosivo), rovinando le feature semantiche pre-addestrate del backbone RoBERTa.*

---

### 🎉 Complimenti! Hai completato il Deep Dive Master di AURA V10.
Ora sei padrone della teoria (Transformer), dell'evoluzione (Encoder), dell'architettura (TS-MHA), dell'ottimizzazione (Loss) e della valutazione (Metriche). Una preparazione impeccabile per qualsiasi discussione o analisi tecnica.
