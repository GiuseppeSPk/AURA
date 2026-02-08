# AURA: Relazione Tecnica Finale
**Tecnhnologies for Multimodal Data Representation and Archives (2025/2026)**

**Team**: Giuseppe Spicchiarello, Mahmoud Hosseini Pour  
**Data**: 28 Gennaio 2026  
**Versione Finale**: V10.2

---

## 1. Introduzione Onesta

Quando abbiamo iniziato questo progetto, l'obiettivo era semplicissimo: *"Vediamo se BERT riesce a riconoscere la tossicità."* 

Pensavamo di fare un semplice fine-tuning su OLID e consegnare. Poi, guardando i primi risultati, ci siamo accorti di un problema: il modello non distingueva la *tossicità* dall'uso di *termini volgari*. Frasi come *"Questo film fa cagare"* venivano flaggate come tossiche, quando in realtà sono solo volgari ma non offensive verso una persona.

Questo ci ha portato in un viaggio molto più complesso del previsto, che ci ha insegnato una lezione fondamentale: **i numeri alti non significano nulla se il modello bara.**  
Abbiamo scoperto che persino i modelli più avanzati sono davvero pigri, il primo pattern che individuano è quello che gli permette di fare meno sforzo per minimizzare l'errore e, se non si è cauti, si rischia di ottenere risultati apparentemente solidi ma non effettivamente allineati con il task.

Questo documento delinea il percorso evolutivo del progetto AURA: le iterazioni tecniche, le fasi di ottimizzazione e le scelte architetturali che hanno portato alla configurazione finale Gold Standard. 

---

## 2. Il Problema di Partenza

### 2.1 Cosa Non Funziona nei Modelli Standard?

I modelli di toxicity detection attuali soffrono di quello che abbiamo chiamato **"Domain Brittleness"**. In pratica:
- Un modello addestrato (o a cui viene fatto fine-tuning) su Twitter crolla quando lo testi su Wikipedia.
- Un modello che vede la parola "idiota" la classifica sempre come tossica, anche in frasi come *"Ha detto che sono un idiota"* (che è un report, non un insulto diretto).

Il problema di fondo è che questi modelli imparano **legami statistici**, non **intenzioni**.

### 2.2 La Nostra Ipotesi (AURA)

Ci siamo chiesti: *"Esiste qualcosa di più stabile delle parole per identificare la tossicità?"*

La risposta ci è arrivata studiando due concetti:
1. **Perspectivism (Prof. Basile)**: La necessità di preservare le sfumature personali e soggettive degli annotatori si conferma di fondamentale importanza quando si analizzano le emozioni. La tossicità è soggettiva. Due annotatori possono non essere d'accordo. Invece di forzare un'etichetta binaria, possiamo usare le *emozioni* (in particolare rabbia e disgusto) come proxy più universale.
2. **Ispirazione dalla linguistica degli eventi**: Leggendo *"La rappresentazione linguistica degli eventi e la loro annotazione nei testi"* (Sprugnoli, 2024), l'osservazione che gli eventi possono essere classificati in base alla loro natura e alla loro funzione, ha ispirato uno degli obiettivi del progetto: individuare i casi di reporting per declassificarli come non tossici.* Una frase tossica *enunciata direttamente* è un evento ben diverso da una frase tossica *riportata*. Questa intuizione ci ha condotti a creare il task "Reporting Detection" come quarta testa del modello.
**Disclaimer**: I contenuti del libro della professoressa Sprugnoli sono stati fonte di sola ispirazione per il progetto, non sono stati plagiati nè utilizzati se non per trarne libero spunto ai fini della ricerca accademica. (Si ringrazia la professoressa per il prezioso contributo).

**L'ipotesi AURA**: Combinando l'analisi delle emozioni con la consapevolezza del framing linguistico, possiamo creare un modello più stabile e robusto.

---

## 3. Il Nostro Percorso (Da V1 a V10.2)

### 3.1 Fase 1: La Baseline e i Primi Dubbi (V1-V2)

Abbiamo iniziato con l'approccio più lineare possibile: un modello `bert-base-uncased` con una singola testa di classificazione (**V1**). Abbiamo scelto il dataset **OLID** (Offensive Language Identification Dataset) perché è il "gold standard" del settore. Tecnicamente, si tratta di un corpus di **14.100 tweet** annotati in modo gerarchico (Livello A: lingua offensiva vs. non offensiva; Livello B: offesa mirata vs. generica; Livello C: bersaglio individuale, di gruppo o altro). È un benchmark consolidato e rappresentava il punto di partenza perfetto per capire se BERT fosse in grado di distinguere un insulto reale da una semplice frase colloquiale.

L'idea iniziale era: *"Facciamo un fine-tuning su OLID e la consegna è fatta."* 

**Risultato V1**: F1 **~0.72**.
Sebbene il risultato fosse in linea con i benchmark standard, test qualitativi approfonditi (stress test su frasi volgari ma non offensive) hanno rivelato limiti strutturali nella capacità di generalizzazione del modello.

**Il Pivot verso il Multi-Task (V2)**:
Il test della V1 ha rivelato che il modello era "pigro": non distingueva la vera tossicità dall'uso di termini volgari. Una frase come *"Cazzo, che freddo che fa oggi!"* veniva flaggata come offensiva tanto quanto un attacco personale. Per questo siamo passati alla **V2**, integrando il dataset **GoEmotions**. È qui che inizia il nostro cammino verso il **Multi-Task Learning (MTL)**. L'ipotesi era che insegnando al modello a riconoscere rabbia e disgusto, lo avremmo "educato" a ignorare la semplice volgarità per concentrarsi sull'intento aggressivo.
**Problemi riscontrati**:
1. **La "Pigrizia" Statistica e la Trappola del Neutral**: Il modello ha subito mostrato la tendenza a scegliere la via di minor resistenza. Per il task di emozione, abbiamo utilizzato il dataset **GoEmotions** (HuggingFace), che originariamente conta 27 categorie. Per rendere il segnale più forte, abbiamo mappato queste categorie sulla tassonomia di **Ekman** (6 emozioni base), aggiungendo però una **settima classe esplicita per il "Neutral"**. 
   - **Il problema**: Circa il 30% dei campioni era "Neutral". Il modello ha imparato che, nel dubbio, predire "Neutral" abbassava drasticamente la loss senza dover fare lo sforzo di distinguere tra Rabbia e Disgusto. Questo portava a un'accuratezza "gonfiata" ma a una capacità di generalizzazione nulla (overfitting).
2. **Integrità del Segnale (No Pseudo-Labels)**: Inizialmente, si era ipotizzato di "accoppiare" i task assegnando un'etichetta automatica di "Non-Tossico" a tutti i campioni di GoEmotions (pseudo-labeling). Ci siamo accorti però che GoEmotions (estratto da Reddit) contiene frasi emotivamente cariche che sono *intrinsecamente tossiche*. Forzare il modello a vederle come sicure (`tox=0`) introduceva molto rumore, "confondendo" il confine di decisione. Abbiamo quindi scelto un approccio di **Interleaved Single-Task Batches**: ogni batch contiene campioni di un solo tipo (es. solo OLID o solo GoEmotions). I task mancanti vengono marcati con un valore di `ignore_index` (-1), così che il gradiente della tossicità non venga minimamente influenzato dal rumore dei dati emotivi e viceversa. Questa "purezza del gradiente" è stata la chiave per stabilizzare il training, portandoci da un F1 di 0.71 a circa 0.73-0.74 nelle prime versioni intermedie, ma soprattutto volevamo che il modello imparasse concetti e non solo rumore statistico.

### V3: L'Inizio dell'Era Multi-Task
Con la **V3** abbiamo consolidato il setup: `bert-base-uncased` con teste parallele per Tossicità ed Emozione, usando pesi statici (Inverse Frequency) per bilanciare le classi. 

**Risultato V3**: F1 **~0.72**. 

Sebbene avessimo superato la baseline, l'analisi qualitativa ci ha dato un colpo di freddo: il modello era pesantemente "biassato". Prediceva quasi sempre l'emozione "Neutral" perché la mole di dati neutri dominava i gradienti, e i pesi statici non riuscivano a dare abbastanza importanza alle emozioni rare come Rabbia e Disgusto. 

Eravamo a un bivio: aggiungere più dati (che ancora non avevamo) o provare a "forzare" il modello con la matematica. Abbiamo scelto la seconda strada.

---

### 3.2 Fase 2: La Trappola Matematica (V4-V7)

Frustrati dai "limiti" della V3, abbiamo pensato che la soluzione fosse puramente algoritmica. Se i pesi statici non bastavano, avremmo usato pesi dinamici e intelligenti. Abbiamo implementato un arsenale matematico avanzato:
- **Focal Loss** (Lin et al., 2017): Per affrontare lo sbilanciamento delle classi, abbiamo implementato questa funzione di perdita come chiave potenziale per l'ottimizzazione. Invece di usare pesi statici, la Focal Loss permette di pesare dinamicamente gli esempi in base alla difficoltà di classificazione.
  
  **Il limite della Focal Loss**: Ben presto però ci siamo accorti che, sebbene la Focal Loss risolvesse lo sbilanciamento *interno* di ogni task (Easy vs Hard), non risolveva la "guerra" tra i task stessi. Nel nostro sistema Multi-Task, i gradienti delle emozioni e della tossicità "lottano" per la precedenza. Cercare di bilanciarli a mano (es. *"pesiamo la tossicità 1.0 e l'emozione 0.5"*) era un incubo di tentativi ed errori che non portava mai a un equilibrio stabile. Pertanto abbiamo cercato una soluzione matematica che aggirasse il problema.

- **Kendall Uncertainty Loss** (Kendall et al., 2018): Questo framework metodologico è stato adottato per gestire il bilanciamento dinamico dei task e ridurre l'impatto del rumore intrinseco nei dati di addestramento.
  
  **La logica matematica**: Per ogni task $i$, la funzione di perdita diventa $\mathcal{L}_{total} = \sum \frac{1}{2\sigma_i^2}\mathcal{L}_i + \log\sigma_i$. 
  - **Gestione del Rumore e dello Sbilanciamento**: Questa formula è stata fondamentale per il task delle emozioni. Nel nostro dataset (V10.2), su oltre 57.000 campioni, il **Neutral** domina con **31.446** occorrenze, mentre la **Rabbia** (**17.771**) e soprattutto il **Disgusto** (**4.053**) sono decisamente più rari e portano con sé un forte rumore di annotazione (la soggettività del perspectivism). Il termine $\sigma_i^2$ permette al modello di "assorbire" questo rumore: se un task è troppo caotico o sbilanciato in un batch, il modello aumenta l'incertezza $\sigma_i$, "abbassando il volume" di quel task per non rovinare le feature dell'encoder, ma permettendo comunque di imparare i pattern più chiari.

  **La nostra implementazione (V10.2)**: Rispetto alla versione standard del paper, abbiamo introdotto una modifica per la stabilità numerica. Invece di usare la funzione esponenziale per $\sigma$ (che portava a esplosioni di gradienti durante il fine-tuning di RoBERTa), abbiamo usato una **Softplus** per derivare la varianza. Questo garantisce che il peso sia sempre positivo e che la transizione durante l'apprendimento delle incertezze sia fluida, permettendo al task "ancora acerbo" (Reporting) di non distruggere le performance consolidate sul task principale (Toxicity).

**Risultato V4**: Abbiamo ottenuto un picco di **F1-Validation di 0.78**. Inizialmente abbiamo festeggiato: sembrava che la potenza della Kendall Loss e della Focal Loss avesse finalmente sbloccato le prestazioni del modello.

**La delusione della V4 (L'illusione dell'F1)**:
Nonostante i festeggiamenti iniziali, ci siamo accorti che qualcosa non tornava. Mentre il validation set segnava 0.78, sul training set il modello arrivava a **0.95**. Era un segnale chiarissimo: il modello non stava imparando a capire il linguaggio, stava semplicemente **memorizzando** i dati. 

Abbiamo capito che la flessibilità della Kendall Loss era diventata un'arma a doppio taglio: il modello la usava per "nascondere" i suoi errori aumentando l'incertezza sui casi difficili, finendo per imparare a memoria le frasi invece di generalizzare. Se la frase non era identica a una già vista, il modello falliva.

**I tentativi falliti (V5-V7)**: prima di capire che il problema era nei dati, abbiamo passato un sacco di tempo a smanettare con i parametri:
- **V5**: Abbiamo provato **DistilBERT** per avere un modello più piccolo e meno incline a memorizzare, ma abbiamo ottenuto solo prestazioni scarse.
- **V6**: Abbiamo alzato il **Dropout al 50%**, ma il training è diventato lentissimo e instabile. Troppi neuroni spenti impedivano al modello di catturare anche i pattern più semplici.
- **V7**: Abbiamo provato con un **Weight Decay** molto cattivo, ma il modello ha iniziato a "dimenticare" anche le cose giuste.

Eravamo a un vicolo cieco. La matematica non poteva sostenere una struttura pericolante, l'unica opzione plausibile rimasta era il **data engineering**.
---

### 3.3 Fase 3: Il Ritorno alla Realtà (V8-V9)

Con la consapevolezza che nessuna ottimizzazione algoritmica avrebbe risolto i problemi strutturali, abbiamo spostato il focus: meno "tuning" dei parametri e più cura nella selezione dei campioni. La Fase 3 è stata la nostra **rivoluzione data-centric**.

**La Strategia V8 (Data Engineering)**:
Abbiamo deciso di bilanciare i task non più solo con la loss, ma alla radice. 
1. **Deduplicazione (Il problema dell'F1 che non convinceva)**: Anche se i numeri della validazione erano buoni, quando provavamo il modello con frasi scritte da noi, sbagliava spesso. Ci siamo chiesti: "Ma se l'F1 è alto, perché il modello sembra così stupido?". Controllando meglio i file, abbiamo trovato la risposta: c'erano dei duplicati tra training e validation set. Il modello non stava indovinando, stava solo ricordando risposte che aveva già visto. Pulire i dati è stato il primo passo per avere una valutazione onesta del modello.
2. **Il Collo di Bottiglia dei Dati (Class Imbalance)**: Analizzando la composizione, ci siamo resi conto dello squilibrio: **OLID** (~12.000 campioni per la tossicità) era "affogato" da **GoEmotions** (~57.000 campioni per le emozioni). I gradienti del task principale erano minoritari.
   
3. **SST-2 come "Ancora Semantica"**: L'analisi degli errori ha rivelato un pattern critico: il modello confondeva la negatività generica per tossicità. Frasi come *"I hate traffic"* venivano classificate come tossiche. Per risolvere, abbiamo integrato il dataset **SST-2** (Stanford Sentiment Treebank, ~72.000 campioni). L'obiettivo era insegnare al modello che **Negatività $\neq$ Tossicità**: SST-2 contiene migliaia di recensioni negative di film, dove la negatività è rivolta a oggetti e non a persone.
4. **Capping dei Task Ausiliari**: Nonostante l'utilità di SST-2 e GoEmotions, la loro mole (oltre 130.000 campioni totali) rischiava di "affogare" il segnale della Tossicità. Abbiamo quindi applicato un **sampling limitato**: SST-2 e GoEmotions sono stati cappati a **20.000 campioni ciascuno**, mentre Toxicity è rimasta integra (~12.000). In questo modo, i gradienti di ogni task contribuivano in modo bilanciato all'aggiornamento dell'encoder. 
   
   **Nota (Work in Progress)**: Siamo consapevoli che il **capping dei dati è il limite principale** del nostro approccio attuale. Come segnalato dal professor Basile, stiamo eseguendo esperimenti comparativi per verificare se un modello più grande (es. `roberta-large`) fine-tuned sull'intera concatenazione del dataset (senza capping) possa portare a performance superiori. I test preliminari sono in corso e verranno discussi successivamente.
5. **La scelta della semplicità: Hate Speech sì o no?**: All'inizio l'idea era quella di aggiungere una quarta testa per l'**Hate Speech**, pensando che "più task = modello più intelligente". Però, ragionandoci meglio, abbiamo capito che avremmo solo aggiunto rumore. L'Hate Speech e la Tossicità sono concetti così vicini che il rischio di creare ambiguità era altissimo. Invece di sovraccaricare il modello con task ridondanti, abbiamo scelto la strada della minimalità: meglio un modello più leggero e focalizzato sull'obiettivo principale (OLID e il rilevamento di tossicità implicita nel testo) che uno inutilmente complicato da segnali quasi identici.

**Risultato V8**: F1 **0.77**.
Il punteggio era simile a quello della V4 "barante", ma questa volta l'Overfitting Gap era crollato dal 18% al **3.2%**. Il modello era finalmente onesto.

**Da V8 a V9: Meno è Meglio**
Con la **V9** abbiamo fatto pulizia. Abbiamo abbandonato definitivamente l'idea dell'Hate Speech perché avevamo capito che, in quel momento, avrebbe solo "confuso" il mazzo dei gradienti senza portare vantaggi reali. Abbiamo preferito concentrare tutta la potenza del modello su tre fronti chiari e distinti: **Tossicità**, **Emozioni** (che abbiamo espanso a 7 classi per essere più precisi) e **Sentiment**. È stato lo step necessario per capire che la strada giusta non era aggiungere task a caso, ma togliere le ambiguità per far respirare l'encoder.

Tuttavia, rimaneva un **limite strutturale**: tutti i task passavano attraverso lo stesso encoder BERT senza specializzazione. Pattern utili per un task (es. l'ironia nel Sentiment) potevano "inquinare" le rappresentazioni di un altro (es. la tossicità diretta). Per superare questo collo di bottiglia, abbiamo progettato l'architettura **Task-Specific Attention** della V10.

---

### 3.4 Fase 4: La svolta dell'architettura (V10)
Siamo arrivati alla conclusione che nessuna funzione di perdita, per quanto intelligente, poteva risolvere un problema che stava alla base: la struttura del modello. Il problema non era come bilanciare i task, ma il fatto che i task "si guardavano" tra loro in modo confuso.

Abbiamo deciso di cambiare rotta completamente, cercando un modo per **separare le rappresentazioni**.

#### 3.4.1 Il Cambio di Backbone: Perché RoBERTa?
Abbiamo abbandonato BERT-base per passare a **`twitter-roberta-base`**. Questa scelta è stata dettata da considerazioni pratiche legate al dominio del nostro task:
1. **Dati di Pre-training**: RoBERTa è stata addestrata su dati social (154 milioni di tweet). Capisce il gergo, le abbreviazioni e lo stile informale molto meglio del BERT standard (addestrato su Wikipedia e libri).
2. **Robustezza BPE**: Il tokenizer BPE (Byte-Pair Encoding) di RoBERTa non si arrende davanti alle parole mascherate. Riesce a ricomporre il significato di *"f**k"* o *"st*pido"* analizzando i byte, lì dove BERT vedeva solo un segnale sconosciuto ([UNK]).

#### 3.4.2 Il vero salto di qualità: Task-Specific Multi-Head Attention (TS-MHA)
Abbiamo implementato questa soluzione applicando il principio della **Multi-Head Attention**, in particolare il concetto di **"Redundancy"**: invece di avere un singolo meccanismo di attenzione condiviso, abbiamo integrato **4 blocchi di Multi-Head Attention paralleli e specializzati** (uno per ogni task).

Questo ci ha permesso di guardare l'intera frase e di isolare quello che serve per ogni task, ottenendo un risultato finalmente pulito.

Abbiamo usato la metafora degli **"8 Investigatori"**:
- Ogni blocco di attenzione ha 8 teste. In quello della Tossicità, per esempio, una testa può imparare a cercare il soggetto (*"Tu"*, *"Voi"*), un'altra i verbi aggressivi, un'altra ancora le negazioni. 
- Questi "investigatori" parlano tra loro (Self-Attention) e creano una rappresentazione del testo specifica per quel problema.

#### 3.4.3 Il Ruolo del Reporting: "POS Eventive Shades"
Per il task di Reporting, ci siamo spinti oltre la semplice ricerca di virgolette. Ispirandoci alla teoria di Sprugnoli, abbiamo cercato di far apprendere al modello le **"sfumature eventive"** delle parti del discorso. 

**Dataset Sintetico e Pseudo-Soggettività**: Non esistendo un dataset pubblico per il Reporting Detection, abbiamo generato un **dataset sintetico** utilizzando **3 Large Language Models diversi** (GPT-OSS 120B, Claude 4.5 (thinking), Gemini 3 Pro (High)). L'obiettivo era simulare il **perspectivism** anche nei dati sintetici: ogni modello ha generato varianti di frasi tossiche (dirette) e le loro versioni "riportate" (citate). Combinando le annotazioni dei 3 LLM, abbiamo ottenuto una **pseudo-soggettività** che riflette diverse "visioni" del concetto di reporting, riducendo il bias di un singolo modello generativo. Il dataset finale contiene ~1000 campioni unici dopo deduplicazione aggressiva.

Il modello deve distinguere tra l'evento come *azione* (*"Mi ha insultato"*) e l'evento come *oggetto riportato* (*"L'insulto è stato riportato"*). La nostra testa di attenzione dedicata (in teoria) impara a riconoscere gli "ancoraggi di reporting", quei verbi e sostantivi che segnalano una transizione funzionale da una dichiarazione a una citazione.

#### 3.4.4 Ottimizzazione Gerarchica
Abbiamo implementato una strategia di ottimizzazione gerarchica per stabilizzare il training multi-task:
1. **Livello Micro**: Usiamo la **Focal Loss** ($\gamma=2.0$) per Toxicity e Sentiment per "zittire" gli esempi facili e forzare il modello a guardare solo i **Hard Negatives** (frasi aggressive ma non tossiche).
2. **Livello Macro**: Usiamo la **Kendall Uncertainty** per bilanciare i task tra loro. In V10.2 abbiamo sostituito la funzione esponenziale standard con una **Softplus**, rendendo il modello molto più stabile e immune alle esplosioni di gradienti durante lo sblocco graduale dei pesi di RoBERTa.

**Progressive Unfreezing (Stabilizzazione delle Teste)**:  
Una scelta tecnica fondamentale per evitare il **Catastrophic Forgetting** è stata l'implementazione del **congelamento progressivo dell'encoder**:
- **Epoca 1**: RoBERTa è completamente **congelato** (`requires_grad=False` per tutti i parametri dell'encoder). Solo le 4 teste Task-Specific MHA e i classification heads vengono addestrati. Questo permette alle teste di "imparare il loro compito" senza rovinare i pesi pre-trained.
- **Epoche 2-15**: RoBERTa viene **sbloccato** (`requires_grad=True`), ma con un learning rate molto conservativo (1e-5 vs 5e-5 delle teste). L'encoder ora può adattarsi finemente ai nostri task senza dimenticare la conoscenza linguistica appresa su 154M tweet.

Questa strategia è stata cruciale: nei test preliminari senza freezing iniziale, l'encoder "collassava" nelle prime epoche perché i gradienti caotici delle teste random lo destabilizzavano. Con il freezing dell'Epoca 1, le teste si allineano prima, e poi l'encoder può adattarsi in modo controllato.

Inoltre, abbiamo implementato i **Differential Learning Rates** e il **Learning Rate Scheduling (Linear Warmup)** proprio per gestire la diversa velocità di apprendimento tra il pretrained backbone (un "Saggio" che richiede LR bassi) e le teste random (gli "Apprendisti" che devono imparare velocemente), evitando il rischio di *Catastrophic Forgetting* discusso a lezione.

---

### 3.5 Fase 5: Gli ultimi ritocchi (V10.2 - 28 Gennaio 2026)

Tutto sembrava pronto. Il training girava bene e i risultati erano ottimi. Però quel **F1 di 1.0** sul task "Reporting" continuava a non convincerci: in statistica, se tutto è perfetto, di solito c'è un errore nascosto. Abbiamo deciso di controllare meglio, e abbiamo trovato le ultime due grane.

#### 3.5.1 Il problema dei dati (Template Leakage)
Siamo andati a spulciare i CSV del dataset "Reporting" e abbiamo iniziato a confrontare le frasi una per una. È stata una brutta sorpresa scoprire che i nostri esempi erano pieni di quasi-duplicati. 
Per esempio:
- Train: *"You are stupid"*
- Validation: *"you are stupid"* (cambiava solo la maiuscola)

Il modello non stava imparando la logica del reporting, stava solo ricordando a memoria le frasi. È stato frustrante dover cancellare metà dei dati a poche ore dalla consegna, ma non avevamo scelta se volevamo un lavoro serio.

**Soluzione**: Abbiamo scritto uno script di deduplicazione "fuzzy" aggressiva. Abbiamo normalizzato tutto il testo e rimosso ogni sovrapposizione semantica tra train e validation. Siamo scesi a **~1000 campioni unici**, ma finalmente avevamo un test onesto.

#### 3.5.2 Il bug dei gradienti (Phantom Gradients)
Mentre sistemavamo i dati, ci siamo accorti che ogni tanto il training "impazziva". Dopo un bel po' di debugging, abbiamo trovato il colpevole nella Kendall Loss. 

Quando in un batch non c'erano esempi per un certo task (tipo il Reporting), il modello cercava comunque di aggiornare l'incertezza $(\sigma)$ per quel task. Questo creava dei gradienti strani che sballavano tutto l'encoder.

**Soluzione V10.2**: Il fix finale è stato l'introduzione della **Task Mask**. 
```python
# V10.2: Il colpo di grazia al bug
term = precision * loss + 0.5 * softplus(log_var)
total += term * mask[i]  # La maschera blocca i gradienti superflui
```
Da quel momento, il training è diventato una linea retta e stabile verso la convergenza.

#### 3.5.3 Configurazione Finale del Training (V10.2)
Per trasparenza, riportiamo la configurazione esatta utilizzata per il training finale su **Kaggle T4 GPU**:

| Parametro | Valore | Giustificazione |
|-----------|--------|-----------------|
| **Batch Size** | 16 | Limite della VRAM (16GB). |
| **Gradient Accumulation** | 4 | Effective batch = 64, più stabile senza aumentare la memoria. |
| **Epoche** | 15 | Con Early Stopping (patience=5) attivato dopo 14 epoche. |
| **Learning Rate (Encoder)** | 1e-5 | Estremamente conservativo per preservare i pesi pre-trained di RoBERTa. |
| **Learning Rate (Task Heads)** | 5e-5 | 5x più veloce dell'encoder: le teste sono random e devono convergere rapidamente. |
| **Weight Decay** | 0.01 | Regolarizzazione L2 leggera. |
| **Focal Loss $\gamma$** | 2.0 | Standard per dataset sbilanciati (Lin et al., 2017). |
| **Dropout** | 0.3 | Più alto del default RoBERTa (0.1) per MTL, previene overfitting sulle teste. |
| **Max Sequence Length** | 128 token | Sufficiente per tweet e post brevi. |

---

## 4. Risultati Finali (V10.2)

| Task | Validation F1 | Class-Specific Detail (Epoch 9) |
|------|---------------|-------------------------------|
| **Toxicity** | **0.7572** | [Neg] F1: 0.81 | [Pos] F1: 0.70 |
| Reporting | 1.00 | P: 1.00 | R: 1.00 |
| Sentiment | Auxiliary | Non tracciato (supporto semantico) |
| Emotion | Auxiliary | Non tracciato (supporto perspectivism) |

### 4.2 Confronto con le Versioni Precedenti

| Versione | F1 Score | Overfitting Gap | Verdetto |
|----------|----------|-----------------|----------|
| V3 (Baseline) | 0.72 | ~5% | Funzionale ma cieco |
| V4 (Kendall Naive) | **0.78** | **18.4%** | **Scartato** (memorizzazione) |
| V8 (Data Balanced) | 0.77 | 3.2% | Stabile ma architettura debole |
| **V10.2 (Final)** | **0.7572** | **~2%** | **Gold Standard** |

**Nota importante**: Il nostro F1 finale (0.7572) è leggermente inferiore al "picco" di V4 (0.78). Questa è una **scelta consapevole**. Preferiamo un modello che generalizza onestamente a uno che bara tramite overfitting.

### 4.3 Technical Deep Dive: Analisi Matematica del Training

L'esame della V10.2 rivela pattern tecnici fondamentali che giustificano le nostre scelte architetturali:

#### 1. Bilanciamento delle Classi (Toxicity Breakdown)
A differenza di modelli "biasati", la V10.2 mostra una notevole capacità di catturare la classe positiva (Tossica):
- **Toxicity [Pos] (Epoca 9)**: Precision 0.63 | **Recall 0.79**. 
- Il fatto che la Recall (0.79) sia superiore alla Precision (0.63) indica che abbiamo configurato il modello per essere un filtro "prudente": preferiamo qualche falso positivo (frase non tossica flaggata) piuttosto che lasciar passare un insulto reale (falso negativo).

#### 2. L'Evoluzione dei Pesi di Kendall ($\sigma$)
Monitorare i parametri apprendibili dell'incertezza omoscedastica è stato fondamentale. Ecco come sono cambiati i pesi ($ \log \sigma^2 $) tra l'Epoca 1 e l'Epoca 9:

| Task | Peso Iniziale (Ep. 1) | Peso Finale (Ep. 9) | Trend e Interpretazione |
|------|-----------------------|----------------------|-------------------------|
| **Toxicity** | **1.435** | **1.305** | **In calo**: Il modello ha ridotto l'incertezza, aumentando l'importanza del gradiente. |
| **Emotion** | 1.431 | 1.300 | **In calo**: Stesso trend, il segnale emotivo è diventato una base solida. |
| **Sentiment** | 1.460 | 1.760 | **In salita**: Incertezza aumentata. Il sentiment era troppo rumoroso rispetto al task primario. |
| **Reporting** | 1.437 | 1.591 | **In salita**: Il modello ha capito che il task era "risolto" e ha ridotto il suo impatto. |

#### 3. Rationale dell'Early Stopping
Il training si è interrotto all'**Epoca 14** (Trigger: Patience 5/5) perché il miglior F1 era stato raggiunto all'**Epoca 9**. Tra l'epoca 9 e la 14, abbiamo osservato la **Loss di Toxicity calare ancora** (da 0.39 a 0.29) mentre l'F1 di validazione restava piatto. Questo è il segnale classico di un inizio di memorizzazione (overfitting): il modello stava "perfezionando" la loss sui dati visti senza più guadagnare capacità di astrazione. Fermarsi all'epoca 9 è stata la scelta corretta per preservare la generalizzazione.

---

## 5. Analisi degli Errori

### 5.1 Dove il Modello Funziona Bene (La Vittoria sulla V1)

- **Insulti diretti**: *"Sei un idiota"* → Correttamente flaggato.
- **Sentiment negativo non tossico**: *"Odio il lunedì"* → Correttamente ignorato.
- **Il trionfo sulle negazioni**: *"Non penso che tu sia stupido"* → La V10, grazie alle 8 teste di attenzione, "vede" il legame tra *non* e *stupido* e lo classifica come **Non-Tossico**, lì dove la baseline crollava.
- **Citazioni**: *"Ha detto 'sei un idiota'"* → La testa Reporting riconosce il framing e neutralizza l'insulto.

### 5.2 Dove il Modello Fallisce Ancora (La Frontiera Finale)

- **Odio implicito e deumanizzazione**: *"Gente come te dovrebbe stare allo zoo"*. Qui il modello fallisce. Non ci sono parole "tossiche" nel dizionario standard, e la deumanizzazione è un concetto semantico troppo profondo per RoBERTa senza una base di conoscenza esterna.
- **Sarcasmo sottile**: *"Complimenti, sei proprio un genio!"* (detto con intento offensivo). Senza il tono di voce o il contesto della conversazione, il modello vede solo un complimento e un sentimento positivo.

Questi limiti confermano che la tecnologia attuale, per quanto avanzata, fatica ancora con l'**inferenza pragmatica**. Sarebbe interessante, in futuro, integrare modelli di *Reasoning* (come quelli basati su grafi di conoscenza) per catturare questi casi limite.

---

## 6. Riflessioni Finali

### Cosa ci portiamo a casa
1. **Non fidarti troppo dei numeri**. Puntare a un F1 altissimo senza guardare l'overfitting è inutile. Abbiamo imparato che un 0.75 "onesto" vale molto di più di un 0.80 ottenuto barando o memorizzando i dati.

2. **L'architettura conta più dei parametri**. Abbiamo passato giorni a cercare la formula magica per la loss, quando la vera svolta è stata dare al modello "occhi diversi" con la Task-Specific Attention. Se la struttura del modello è sbagliata, non c'è parametro che tenga.

3. **I dati sono la parte più difficile**. Quel "1.0" sul Reporting è stato un grande insegnamento. I dati sintetici possono ingannare facilmente se non si fa una pulizia profonda e manuale.

4. **Il contesto è tutto**. Capire se una frase è tossica o se è solo una citazione non è questione di parole chiave, ma di come quelle parole sono messe insieme.

### Cosa Faremmo Diversamente

Se potessimo tornare al primo giorno di laboratorio:
- **Struttura Subito**: Implementeremmo la Task-Specific Attention come prima mossa, invece di considerarla un'opzione avanzata.
- **Data-First**: Passaremmo molto più tempo a pulire e analizzare i dataset prima di lanciare ore di training su Kaggle.
- **Semplicità**: Non cercheremmo soluzioni iper-complesse (come la Kendall originale) senza prima aver capito i limiti della baseline.
- **Unione dei Dataset**: Come riflessione finale, seguendo anche uno spunto del professor Basile, probabilmente invece di limitarci al sampling dei dati ausiliari (GoEmotions, SST-2) per proteggere OLID, avremmo potuto sperimentare con la concatenazione totale dei dataset su un encoder ancora più capace. Questo avrebbe potuto sbloccare performance ancora migliori sfruttando ogni singolo campione a disposizione.

---

## 7. Allegati

1. **AURA_V10_KAGGLE.ipynb** — Notebook completo (Kaggle-ready)
2. **aura-v10-data/** — Dataset pulito e deduplicato
3. **AURA_README.md** — Documentazione tecnica completa

---

*Technical Report - Research Release V10.2*
*Grazie per aver letto fin qui.*
