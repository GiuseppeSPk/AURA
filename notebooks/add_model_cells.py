import nbformat as nbf

# Load existing notebook
nb = nbf.read('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_DEFENSE_ANNOTATED.ipynb', as_version=4)

print(f"Current cells: {len(nb.cells)}")

# Helper functions
def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ===== MODEL ARCHITECTURE CELLS =====

# Dataset Classes with full annotations
dataset_code = code("""# Cell 3: Dataset Classes (MTL-Compatible)

# === THEORY: Custom Dataset Architecture ===
#
# PyTorch Dataset Protocol:
#   __init__: Load data into memory (or prepare file handles)
#   __len__: Return total number of samples
#   __getitem__(idx): Return a single sample as a dict
#
# Our Innovation: Each dataset returns a 'task' field:
#   - Task 0: Toxicity
#   - Task 1: Emotion
#   - Task 2: Sentiment
#   - Task 3: Reporting
#
# This allows the collate_fn to create MIXED batches.

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    \"\"\"Base class for all task-specific datasets.
    
    DESIGN PATTERN: Template Method
      - Common logic (tokenization, encoding) in parent
      - Task-specific logic (__getitem__) in children
    \"\"\"
    def __init__(self, path, tokenizer, max_len):
        self.df = pd.read_csv(path)
        self.tok = tokenizer
        self.max_len = max_len
        
    def __len__(self): 
        return len(self.df)
    
    def encode(self, text):
        \"\"\"Tokenize and encode text to fixed-length tensor.
        
        PROCESS:
          1. Tokenize: "I hate traffic" → ["I", "hate", "traffic"]
          2. Convert to IDs: ["I", "hate", "traffic"] → [146, 6123, 7628]
          3. Add special tokens: [0, 146, 6123, 7628, 2]  # 0=<s>, 2=</s>
          4. Pad to max_length: [0, 146, 6123, 7628, 2, 1, 1, 1, ...]  # 1=<pad>
          5. Create attention_mask: [1, 1, 1, 1, 1, 0, 0, 0, ...]
        
        WHY padding='max_length'?
          - GPU requires FIXED tensor sizes for batching
          - Variable-length would require custom CUDA kernels
        
        WHY truncation=True?
          - Some samples exceed 128 tokens (e.g., long Reddit posts)
          - Truncation: Keep first 126 tokens + special tokens
          - Alternative (discarded): Could split into multiple samples
        \"\"\"
        return self.tok(
            str(text),                  # Safety: convert NaN to "nan" string
            max_length=self.max_len,    # Fixed length for GPU efficiency
            padding='max_length',        # Pad short sequences to max_length
            truncation=True,             # Truncate long sequences
            return_tensors='pt'          # Return PyTorch tensors (not lists)
        )

class ToxicityDataset(BaseDataset):
    \"\"\"OLID Toxicity Detection (Binary Classification).
    
    DATA SOURCE: OLID (Offensive Language Identification Dataset)
      - Twitter posts
      - Labels: 0 (NOT offensive), 1 (Offensive)
      - Class imbalance: ~95% NOT, ~5% Offensive
    \"\"\"
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.encode(row['text'])
        return {
            'ids': enc['input_ids'].flatten(),       # Shape: [128]
            'mask': enc['attention_mask'].flatten(), # Shape: [128]
            'tox': torch.tensor(int(row['label']), dtype=torch.long),  # Scalar: 0 or 1
            'task': 0  # Task ID for routing in training loop
        }

class EmotionDataset(BaseDataset):
    \"\"\"GoEmotions (Multilabel Emotion Classification).
    
    DATA SOURCE: GoEmotions (Google Research, 2020)
      - Reddit comments
      - 27 original emotions (reduced to 7 Ekman categories)
      - MULTILABEL: A comment can be [angry=1, sad=1, neutral=0, ...]
    
    CRITICAL FIX: Filter samples with label_sum=0 (no emotions)
      - Original dataset has ~3% samples with ALL zeros
      - These break BCE loss (no positive class to learn from)
      - We remove them during loading and reset_index to avoid gaps
    \"\"\"
    def __init__(self, path, tokenizer, max_len, cols):
        super().__init__(path, tokenizer, max_len)
        self.cols = cols  # ['anger', 'disgust', 'fear', ...]
        
        # STABILITY FIX: Remove samples with no emotion labels
        if 'label_sum' in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df['label_sum'] > 0].reset_index(drop=True)
            removed = before - len(self.df)
            if removed > 0:
                print(f\"   ⚠️ Removed {removed} samples with no emotion labels\")
            
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.encode(row['text'])
        return {
            'ids': enc['input_ids'].flatten(),
            'mask': enc['attention_mask'].flatten(),
            'emo': torch.tensor([float(row[c]) for c in self.cols], dtype=torch.float),  # Shape: [7]
            'task': 1
        }

class SentimentDataset(BaseDataset):
    \"\"\"SST-2 Sentiment Analysis (Binary Classification).
    
    DATA SOURCE: Stanford Sentiment Treebank v2
      - Movie reviews
      - Labels: 0 (Negative), 1 (Positive)
      - Balanced: ~50/50 split
    
    WHY Sentiment for Toxicity?
      - Sentiment teaches POLARITY (positive/negative axis)
      - Toxicity is always negative, but not all negative is toxic
      - Forces the model to learn INTENSITY distinction
    \"\"\"
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.encode(row['text'])
        return {
            'ids': enc['input_ids'].flatten(),
            'mask': enc['attention_mask'].flatten(),
            'sent': torch.tensor(int(row['label']), dtype=torch.long),
            'task': 2
        }

class ReportingDataset(BaseDataset):
    \"\"\"Reporting Event Detection (Binary Classification).
    
    DATA SOURCE: Custom (based on Prof. Sprugnoli's theory)
      - 500 samples (250 OCCURRENCE, 250 REPORTING)
      - Balanced by design
    
    EXAMPLES:
      - OCCURRENCE (label=0): "You are an idiot" (direct insult)
      - REPORTING (label=1): "He said you are an idiot" (quoted insult)
    
    LINGUISTIC MARKERS:
      - Reporting verbs: "said", "claims", "wrote", "mentioned"
      - Quotation marks: \"...\", '...'
      - Attribution: "According to X", "X stated that"
    \"\"\"
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.encode(row['text'])
        return {
            'ids': enc['input_ids'].flatten(),
            'mask': enc['attention_mask'].flatten(),
            'rep': torch.tensor(int(row['is_reporting']), dtype=torch.long),
            'task': 3
        }

def collate_fn(batch):
    \"\"\"Custom collate function for mixed-task batches.
    
    STANDARD PyTorch collate:
      batch = [sample1, sample2, ...]
      return torch.stack([s['ids'] for s in batch])  # Assumes ALL samples have SAME keys
    
    OUR CHALLENGE:
      - Sample 1 (Toxicity): has 'tox', NO 'emo'
      - Sample 2 (Emotion): has 'emo', NO 'tox'
    
    SOLUTION:
      - Stack common keys (ids, mask, task)
      - For task-specific keys, collect ONLY from samples with that task
      - Return None if no samples of that task in batch
    
    EXAMPLE:
      Input batch: [Tox, Tox, Emo, Sent]
      Output:
        'tox': [label1, label2]  # Only 2 toxicity samples
        'emo': [label3]          # Only 1 emotion sample
        'sent': [label4]         # Only 1 sentiment sample
        'rep': None              # No reporting samples in this batch
    \"\"\"
    # Stack common fields (all samples have these)
    ids = torch.stack([x['ids'] for x in batch])
    mask = torch.stack([x['mask'] for x in batch])
    tasks = torch.tensor([x['task'] for x in batch])
    
    # Collect task-specific labels (conditional stacking)
    tox_items = [x['tox'] for x in batch if x['task'] == 0]
    emo_items = [x['emo'] for x in batch if x['task'] == 1]
    sent_items = [x['sent'] for x in batch if x['task'] == 2]
    rep_items = [x['rep'] for x in batch if x['task'] == 3]
    
    return {
        'ids': ids,      # Shape: [batch_size, 128]
        'mask': mask,    # Shape: [batch_size, 128]
        'tasks': tasks,  # Shape: [batch_size]
        'tox': torch.stack(tox_items) if tox_items else None,
        'emo': torch.stack(emo_items) if emo_items else None,
        'sent': torch.stack(sent_items) if sent_items else None,
        'rep': torch.stack(rep_items) if rep_items else None
    }

print('📦 Dataset classes and collate function defined.')
""")

nb.cells.append(dataset_code)

# Model Architecture
model_code = code("""# Cell 4: AURA V10 Model Architecture

# === THEORY RECAP: Why Task-Specific Attention? ===
#
# PROBLEM with shared CLS token:
#   All tasks forced to use SAME 768-dim vector
#   → Negative transfer (Emotion noise confuses Toxicity)
#
# SOLUTION: Parallel Attention Heads
#   Each task gets its OWN attention → feature disentanglement

class TaskSpecificMHA(nn.Module):
    \"\"\"Multi-Head Self-Attention for a single task.
    
    ATTENTION INTUITION:
      - Query: "What am I looking for?"
      - Key: "What do I offer?"
      - Value: "What information do I carry?"
    
    For Toxicity MHA:
      - Query focuses on: pronouns ("You"), insults
    For Reporting MHA:
      - Query focuses on: "said", quotation marks
    
    The MODEL LEARNS what to focus on via backpropagation!
    \"\"\"
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        
        # PyTorch's nn.MultiheadAttention:
        #   - Implements scaled dot-product attention
        #   - Splits hidden_dim into n_heads subspaces automatically
        #   - Example: 768 dims / 8 heads = 96 dims per head
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,   # 768 for RoBERTa-base
            num_heads=n_heads,      # 8 (standard for 768-dim models)
            batch_first=True,       # Input shape: [batch, seq_len, hidden]
            dropout=dropout         # Attention dropout (NOT output dropout)
        )
        
        # LayerNorm: Stabilizes training by normalizing activations
        # Formula: (x - mean) / sqrt(var + eps)
        self.layernorm = nn.LayerNorm(hidden_dim)
        
        # Dropout: Randomly zeros 10% of activations during training
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask):
        \"\"\"
        Args:
            hidden_states: [batch, seq_len, 768] from RoBERTa
           attention_mask: [batch, seq_len] where 1=real token, 0=padding
        
        Returns:
            output: [batch, seq_len, 768] attended representation
            attn_weights: [batch, n_heads, seq_len, seq_len] attention scores
        \"\"\"
        # IMPORTANT: PyTorch MHA uses INVERTED mask semantics
        #   - Our mask: 1=attend, 0=ignore
        #   - MHA expects: True=ignore, False=attend
        key_padding_mask = (attention_mask == 0)
        
        # Self-Attention: Query, Key, Value ALL come from hidden_states
        attn_output, attn_weights = self.mha(
            query=hidden_states, 
            key=hidden_states, 
            value=hidden_states,
            key_padding_mask=key_padding_mask  # Don't attend to padding tokens
        )
        
        # Residual Connection + LayerNorm (Transformer standard pattern)
        # WHY Residual?
        #   - Allows gradients to flow directly through the network
        #   - Prevents "degradation" problem in deep networks
        # WHY LayerNorm AFTER add?
        #   - Post-LN: More stable for pre-trained models
        output = self.layernorm(hidden_states + self.dropout(attn_output))
        
        return output, attn_weights

class AURA_V10(nn.Module):
    \"\"\"AURA V10: RoBERTa + 4 Parallel Task-Specific MHA + 4 Classification Heads.
    
    ARCHITECTURE FLOW:
      Text → RoBERTa → Shared Hidden States →  ┬─ Tox MHA → Pool → Tox Head
                                                ├─ Emo MHA → Pool → Emo Head
                                                ├─ Sent MHA → Pool → Sent Head
                                                └─ Rep MHA → Pool → Rep Head
    \"\"\"
    def __init__(self, config):
        super().__init__()
        
        # === SHARED ENCODER ===
        # Load pre-trained RoBERTa-base from Hugging Face
        # This includes 12 Transformer layers (125M parameters)
        self.roberta = RobertaModel.from_pretrained(config['encoder'])
        hidden = config['hidden_dim']  # 768
        
        # === TASK-SPECIFIC ATTENTION BLOCKS (Parallel) ===
        # Each task gets its OWN attention mechanism
        # Total params: 4 × (8 heads × 96 dims × 3 projections) ≈ 4 × 2.3M = 9.2M params
        self.tox_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])
        self.emo_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])
        self.sent_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])
        self.report_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])
        
        # Dropout before classification (MODULE 3 regularization)
        self.dropout = nn.Dropout(config['dropout'])
        
        # === CLASSIFICATION HEADS ===
        # Simple linear layers (no hidden layers)
        # WHY no hidden layers?
        #   - RoBERTa already has 12 layers of non-linearity
        #   - Adding more risks overfitting on small datasets
        self.toxicity_head = nn.Linear(hidden, 2)       # Binary: [Non-Tox, Tox]
        self.emotion_head = nn.Linear(hidden, config['num_emotion_classes'])  # Multilabel: 7 emotions
        self.sentiment_head = nn.Linear(hidden, 2)      # Binary: [Neg, Pos]
        self.reporting_head = nn.Linear(hidden, 1)      # Binary (as sigmoid): [0, 1]
        
        # === BIAS INITIALIZATION (MODULE 3: Imbalanced Datasets) ===
        # See theory above: Set bias to log-odds of class frequencies
        with torch.no_grad():
            # Toxicity: Prior = [95%, 5%]
            self.toxicity_head.bias[0] = 2.5   # log(0.95/0.05) ≈ 2.94
            self.toxicity_head.bias[1] = -2.5  # log(0.05/0.95) ≈ -2.94
            
            # Other tasks: Balanced or multilabel, no bias init needed

    def _mean_pool(self, seq, mask):
        \"\"\"Masked mean pooling over sequence dimension.
        
        FORMULA:
          pooled = Σ(mask_i × seq_i) / Σ(mask_i)
        
        WHY not just mean()?
          - mean() would include PADDING tokens (garbage values)
          - Masked pooling: Only real tokens contribute
        
       Args:
            seq: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len]
        
        Returns:
            pooled: [batch, hidden_dim]
        \"\"\"
        # Expand mask to match sequence dimensions
        mask_exp = mask.unsqueeze(-1).expand(seq.size()).float()  # [batch, seq_len, hidden]
        
        # Element-wise multiply: zeros out padding tokens
        masked_seq = seq * mask_exp  # [batch, seq_len, hidden]
        
        # Sum over sequence, divide by count of real tokens
        sum_seq = masked_seq.sum(dim=1)  # [batch, hidden]
        count = mask_exp.sum(dim=1).clamp(min=1e-9)  # [batch, hidden], prevent div-by-zero
        
        return sum_seq / count

    def forward(self, input_ids, attention_mask):
        \"\"\"Forward pass: Text → Logits for all tasks.
        
        Args:
            input_ids: [batch, 128] token IDs
            attention_mask: [batch, 128] padding mask
        
        Returns:
            dict: {
                'toxicity': [batch, 2],
                'emotion': [batch, 7],
                'sentiment': [batch, 2],
                'reporting': [batch]  # Note: scalar per sample
            }
        \"\"\"
        # === STEP 1: Shared Encoding ===
        #  RoBERTa processes ALL samples together
        # Output: [batch, 128, 768] (one vector per token)
        shared = self.roberta(input_ids, attention_mask).last_hidden_state
        
        # === STEP 2: Task-Specific Attention (PARALLEL) ===
        # Each MHA focuses on different features in the SAME shared representation
        tox_seq, _ = self.tox_mha(shared, attention_mask)    # [batch, 128, 768]
        emo_seq, _ = self.emo_mha(shared, attention_mask)    # [batch, 128, 768]
        sent_seq, _ = self.sent_mha(shared, attention_mask)  # [batch, 128, 768]
        rep_seq, _ = self.report_mha(shared, attention_mask) # [batch, 128, 768]
        
        # === STEP 3: Pool + Classify ===
        # Reduce [batch, 128, 768] → [batch, 768] via mean pooling
        # Then apply linear classifier
        return {
            'toxicity': self.toxicity_head(self.dropout(self._mean_pool(tox_seq, attention_mask))),
            'emotion': self.emotion_head(self.dropout(self._mean_pool(emo_seq, attention_mask))),
            'sentiment': self.sentiment_head(self.dropout(self._mean_pool(sent_seq, attention_mask))),
            'reporting': self.reporting_head(self.dropout(self._mean_pool(rep_seq, attention_mask))).squeeze(-1)
            # squeeze(-1): [batch, 1] → [batch] for BCE loss compatibility
        }

print('🦅 AURA_V10 model architecture defined.')
print(f'   Total Parameters: ~{125 + 9.2 + 0.01:.1f}M (RoBERTa + MHAs + Heads)')
""")

nb.cells.append(model_code)

# Save
nbf.write(nb, 'C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_DEFENSE_ANNOTATED.ipynb')
print(f"✅ Notebook now has {len(nb.cells)} cells")
print("Added: Configuration, Datasets, Model Architecture")
print("Remaining: Loss Functions, Training Loop, Evaluation")
