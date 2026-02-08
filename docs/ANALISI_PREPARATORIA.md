# AURA: Preliminary Technical Analysis

Per garantire la validità tecnica dell'implementazione ed evitare regressioni algoritmiche, è necessario adottare una strategia di **"Fail Fast, Fix Faster"**. Non basta implementare l'architettura; è fondamentale validare le assunzioni scientifiche alla base di AURA.

Ecco l'analisi dei rischi e il piano d'azione preventivo.

---

## 1. Analisi dei Rischi (Critical Failure Points)

### 🔴 Rischio 1: L'Ipotesi "Ponte Emozionale" è debole
**Il problema**: Assumiamo che "Hate Speech" implichi sempre Rabbia o Disgusto. Ma se il dataset OLID contiene odio "freddo" (calmo, sarcastico) o se GoEmotions etichetta la critica legittima come "Rabbia", il segnale emozionale potrebbe diventare *rumore* invece che aiuto.
**Verifica Pre-Training**: Dobbiamo verificare la correlazione statistica sui dati *prima* di addestrare la rete neurale.

### 🔴 Rischio 2: "Domain Gap" tra Reddit e Twitter
**Il problema**: AURA usa GoEmotions (Reddit, testi lunghi, discussioni) per imparare le emozioni e OLID (Twitter, testi brevi, slang) per la tossicità.
Se i vocabolari sono disgiunti (es. Reddit usa inglese formale, Twitter usa slang), l'encoder BERT imparerà due distribuzioni diverse e non riuscirà a trasferire la conoscenza.
**Verifica**: Calcolo della "Vocabulary Intersection" (sovrapposizione dei termini).

### 🔴 Rischio 3: Il "Buco Nero" della classe Neutra
**Il problema**: Anche separando il Neutro (soluzione che abbiamo scelto), c'è il rischio che il modello impari a classificare tutto ciò che è difficile come "Neutro" per minimizzare la loss, ignorando le sfumature sottili di Rabbia/Disgusto necessarie per AURA.
**Mitigazione**: Monitoraggio specifico della F1-Score sulla classe Neutra durante le prime epoche.

---

## 2. Piano di Azione (La "Checklist di Garanzia")

Dobbiamo inserire una **Fase 0** nel nostro piano di lavoro.

### ✅ Fase 0.1: Validazione Statistica (Data Audit)
Prima di scrivere il loop di training, creiamo un notebook `00_Hypothesis_Validation.ipynb` per:
1.  **Matrice di Correlazione Tossicità-Emozione**:
    *   Prendere un sottoinsieme di dati (se esiste overlap o usando un modello pre-trained veloce) per stimare $P(\text{Toxicity}=1 | \text{Emotion}=e)$.
    *   *Success Condition*: $P(\text{Tox}|\text{Anger}) \gg P(\text{Tox}|\text{Joy})$. Se sono simili, AURA non funzionerà.
2.  **Analisi Overlap Vocabolario**:
    *   Confrontare i top-1000 token di OLID e GoEmotions.
    *   *Success Condition*: >40% di overlap significativo (escluse stopwords).

### ✅ Fase 0.2: Baseline "Onesta"
Molti progetti falliscono perché la baseline è troppo debole e il nuovo modello sembra forte solo per confronto.
*   **Obbligatorio**: Ottimizzare la Single-Task BERT (learning rate, epochs) *al massimo* delle possibilità.
*   Se AURA batte una BERT scarsa, non abbiamo provato nulla. AURA deve battere una BERT forte.

### ✅ Fase 0.3: Debugging del "Gradient Fighting"
In Multi-Task Learning, i gradienti della Loss A (Tossicità) e Loss B (Emozioni) possono andare in direzioni opposte, cancellandosi a vicenda.
*   **Azione**: Implementare il monitoraggio della "Cosine Similarity" tra i gradienti delle due task nelle prime 100 step.
*   Se è sempre negativa, i task sono in conflitto e bisogna cambiare pesi o architettura (es. separare gli ultimi layer dell'encoder).

---

## 3. Modifiche Strategiche al Codice Preesistente

Sulla base di questa analisi, suggerisco di modificare subito:

1.  **`dataset.py`**: Aggiungere un controllo di sanità sui dati in ingresso (es. scartare testi vuoti o < 3 caratteri).
2.  **`trainer.py`**: Aggiungere log separati per `loss_toxicity` e `loss_emotion` (non solo la somma), per vedere se uno dei due domina e "schiaccia" l'altro.
3.  **`config.yaml`**: Aggiungere un parametro `gradient_clipping_per_task` per evitare che un task instabile faccia esplodere tutto.

## 4. Protocollo di Validazione Sistematica

**"Non fidarsi, verificare."**
Invece di lanciare il training finale di 5 ore, lanciamo prima un **"Overfit Test"**:
1.  Prendiamo solo 50 esempi (batch minuscolo).
2.  Addestriamo AURA.
3.  La Training Loss deve andare a ZERO rapidamente.
    *   Se non va a zero: C'è un bug nel codice o nell'architettura.
    *   Se va a zero: Il modello è capace di imparare. Possiamo scalare al dataset completo.

Questo protocollo garantisce l'integrità tecnica e scientifica della ricerca.
