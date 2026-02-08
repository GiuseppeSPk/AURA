# AURA MASTER STUDY GUIDE: Block 1 (Deep Dive Edition)
## Il Cuore del Transformer: Fondamenta, Attention e Analisi Comparativa

Questo blocco analizza in profondità l'architettura introdotta da "Attention Is All You Need" (Vaswani et al., 2017), mettendola a confronto con le tecnologie precedenti e calandola nel contesto specifico dei parametri di AURA.

---

### 1. Il Contesto Storico: Dalle RNN/GRU al Transformer

#### 1.1 Il Problema della Ricorrenza (LSTM/GRU)
Prima del 2017, le reti dominanti erano le **RNN (Recurrent Neural Networks)**, incluse le **LSTM** e le **GRU**.
- **Logica Sequenziale**: Un'RNN elabora il token $t$ solo dopo aver ricevuto l'informazione dal token $t-1$.
- **Vanishing Gradient**: Durante il backpropagation, il gradiente deve attraversare ogni step temporale "moltiplicandosi". Se la frase è lunga (es. 100 parole), il gradiente diventa infinitesimo prima di raggiungere le prime parole.
- **Collo di Bottiglia (Bottleneck)**: Tutta l'informazione della frase deve essere compressa in un unico vettore "hidden state" di raggio fisso.

#### 1.2 Focus: La GRU (Gated Recurrent Unit)
La GRU (2014) è una versione ottimizzata della LSTM che usa due gate principali:
1. **Update Gate ($z_t$)**: Decide quanto dello stato precedente mantenere.
2. **Reset Gate ($r_t$)**: Decide come combinare il nuovo input con la memoria passata.
- **Perché non basta per AURA?** Nonostante i gate, la GRU è ancora sequenziale. Non può calcolare relazioni a lunga distanza con la stessa precisione e velocità di calcolo parallelo del Transformer.

---

### 2. Scaled Dot-Product Attention: Analisi Matematica

In AURA, l'attenzione è il meccanismo che permette di calcolare il peso di ogni parola rispetto alle altre.

#### 2.1 I Tre Attori: Q, K, V (Proiezione in Spazi Latenti)
Ogni token $x$ viene proiettato in tre **spazi latenti** (rappresentazioni astratte) diversi tramite matrici di peso apprese ($W_Q, W_K, W_V$):
- **Query Space ($Q = x \cdot W_Q$)**: Lo spazio latente "interrogativo". Rappresenta "cosa sto cercando" (es. la query di un insulto cerca il suo bersaglio).
- **Key Space ($K = x \cdot W_K$)**: Lo spazio latente "descrittivo". Rappresenta l'identità del token (es. "io sono un pronome", "io sono un aggettivo").
- **Value Space ($V = x \cdot W_V$)**: Lo spazio latente "informativo". Contiene il significato effettivo che verrà trasmesso se il token viene selezionato.

> [!NOTE]
> Un **spazio latente** è una rappresentazione alternativa dei dati che mette in risalto caratteristiche non visibili nell'input originale. In AURA, creiamo tre "versioni" latenti di ogni parola per farle dialogare tra loro.

#### 2.2 La Formula Dettagliata
$$ \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

1. **Il Prodotto Scalare ($QK^T$)**: Misura la similarità tra Query e Key. Per ogni coppia di token $(i, j)$, il punteggio indica quanto il token $i$ deve prestare attenzione al token $j$.
2. **Lo Scaling Factor ($\sqrt{d_k}$)**: Fondamentale. Senza di esso, per $d_k$ alti (come 96 in AURA), i valori del prodotto scalare crescerebbero enormemente. 
   - *Problema*: Valori alti spingono la funzione Softmax in regioni dove il gradiente è quasi zero (Saturazione).
   - *Soluzione*: Dividere per $\sqrt{96} \approx 9.8$ mantiene la varianza dei punteggi vicina a 1, rendendo il training stabile.
3. **Softmax**: Trasforma i punteggi in una distribuzione di probabilità.
4. **Weighted Value**: L'output è la somma dei vettori $V$ pesata per le probabilità del softmax.

---

### 3. Multi-Head Attention (MHA): Parallelismo e Sottospazi

In AURA V10 non calcoliamo l'attenzione una sola volta. Lo facciamo **8 volte in parallelo**.

#### 3.1 La Divisione delle Dimensioni
- **Input Dimension ($d_{model}$)**: 768 (Standard RoBERTa-base).
- **Number of Heads ($h$)**: 8.
- **Head Dimension ($d_k = d_v$)**: $768 / 8 = \mathbf{96}$.

#### 3.2 Perché 8 Teste da 96? (Intuizione Geometrica)
Ogni testa lavora in un **sottospazio a 96 dimensioni**. Questo permette al modello di focalizzarsi su aspetti diversi contemporaneamente:
- Testa 1: relazioni sintattiche (soggetto-verbo).
- Testa 2: riferimenti pronominali (coreference).
- Testa 3: intensificatori emotivi (molto, troppo, per niente).
- ...e così via.

#### 3.3 Il Flusso dei Tensor (Tensor Shapes)
1. **Input**: `[Batch, Seq_Len, 768]`
2. **Proiezione**: `[Batch, Seq_Len, 8, 96]`
3. **Transpose**: `[Batch, 8, Seq_Len, 96]` (necessario per calcolare l'attenzione testa per testa).
4. **Attention**: Risultato `[Batch, 8, Seq_Len, 96]`.
5. **Concatenazione**: Le 8 teste vengono riunite in `[Batch, Seq_Len, 768]`.
6. **Output Projection ($W_O$)**: Una matrice finale $768 \times 768$ che mescola le informazioni delle 8 teste.

---

### 4. Componenti di Stabilizzazione: Add & Norm

Dopo l'attenzione, AURA applica il pattern standard dei Transformer:

#### 4.1 Residual Connection (Skip Connection)
L'output dell'Attention non viene usato da solo, ma viene **sommato all'input originale**: $x + \text{Attention}(x)$.
- **Funzione**: Permette al gradiente di fluire direttamente attraverso i layer senza essere degradato. È la soluzione di ResNet applicata ai testi.

#### 4.2 Layer Normalization
Normalizza i dati lungo la dimensione delle feature (768).
- **Formula**: $\text{LN}(x) = \gamma \frac{x - \mu}{\sigma} + \beta$.
- **In AURA**: Garantisce che le attivazioni abbiano media 0 e varianza 1, prevenendo esplosioni numeriche.

---

### 5. Positional Encoding: Dare un Senso all'Ordine

Il Transformer è "ordine-agnostico" (vede le parole come un set, non come una riga). 
- **Soluzione**: Aggiungiamo un segnale vettoriale che dipende dalla posizione del token.
- **In AURA**: RoBERTa usa **Learned Positional Embeddings**. Il modello "impara" un vettore specifico per la posizione 1, uno per la 2, ecc.
- **Cruciale per Toxicity**: *"Cretino non sono io"* vs *"Non sono io, cretino"*. La posizione dell'insulto cambia il target e l'intento.

---

### AURA BLOCK 1: Transformer Foundations
## Technical Reference Module

1. **"Perché Scaled Dot Product e non solo Dot Product?"**
   *Risp: Per evitare la saturazione del softmax. In alta dimensionalità ($d_k=96$), i prodotti scalari tendono a magnitudo elevate; lo scaling riporta la varianza a livelli ottimali per il gradiente.*

2. **"Come si differenzia l'attenzione di AURA da quella di una GRU con meccanismo di attenzione?"**
   *Risp: In una GRU l'attenzione è spesso applicata solo per "pesare" gli hidden states già calcolati sequenzialmente. Nel Transformer di AURA, l'attenzione È l'architettura stessa (Self-Attention), permettendo parallelismo totale e una risoluzione spaziale del contesto nettamente superiore.*

3. **"Se raddoppiassi le teste a 16 mantenendo 768 dimensioni totali, cosa cambierebbe?"**
   *Risp: La dimensione di ogni testa scenderebbe a 48 ($768/16$). Questo aumenterebbe la diversità delle prospettive (più teste) ma ridurrebbe la capacità espressiva di ogni singola testa (meno dimensioni per catturare pattern complessi).*

---

*Prossimo blocco: Evoluzione degli Encoder (BERT, RoBERTa, DeBERTa).*
