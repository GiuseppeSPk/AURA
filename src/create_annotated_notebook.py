
import json
import os

# Paths
INPUT_NB = r"C:\Users\spicc\Downloads\aura-v10-kaggle.ipynb"
OUTPUT_NB = r"c:\Users\spicc\Desktop\Multimodal\reports\AURA_V10_STUDY_VERSION.ipynb"

# Load Notebook
with open(INPUT_NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to add comments
def annotate_cell_source(source_list, annotations):
    """
    Interleaves code lines with comments or replaces them.
    annotations: dict {line_index: "Comment string"}
    This is a simple approach; for full detailed comments, we might just replace the whole block.
    """
    # Simply returning the new source block defined manually is safer to ensure quality.
    pass

# Define new contents for key cells
# Cell 1: Imports
cell_imports_source = [
    "# === CELL 1: SETUP & FOUNDATIONS ===\n",
    "# THEORETICAL NOTE (Module 1/2): \n",
    "# - Torch/Cuda: The engine for Tensor Calculus.\n",
    "# - Transformers: The library providing the RoBERTa backbone (Module 2).\n",
    "# - Sklearn Metrics: Essential for scientifically valid evaluation (F1, Precision, Recall).\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn  # Neural Network modules (Layers, Activations)\n",
    "import torch.nn.functional as F  # Functional interface (Losses, helper functions)\n",
    "from torch.utils.data import DataLoader, Dataset, ConcatDataset # Data handling logic\n",
    "from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup\n",
    "from tqdm.notebook import tqdm  # Progress bar for training loops\n",
    "from sklearn.metrics import (\n",
    "    f1_score, classification_report, confusion_matrix, \n",
    "    multilabel_confusion_matrix, precision_recall_fscore_support\n",
    ")\n",
    "import pandas as pd  # Data manipulation tables\n",
    "import numpy as np  # Liner Algebra operations\n",
    "import matplotlib.pyplot as plt  # Plotting library\n",
    "import seaborn as sns  # Advanced visualization styling\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')  # Cleaning output from non-critical warnings\n",
    "\n",
    "# === REPRODUCIBILITY (Scientific Method) ===\n",
    "# In science, a result must be reproducible. We lock the 'Seed'.\n",
    "SEED = 42\n",
    "torch.manual_seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "torch.backends.cudnn.deterministic = True  # Force deterministic algorithms on GPU\n",
    "torch.backends.cudnn.benchmark = False     # Disable auto-tuner for consistency\n",
    "\n",
    "# Hardware Check\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'🔧 Device: {device}')\n",
    "if device.type == 'cuda':\n",
    "    print(f'   GPU: {torch.cuda.get_device_name(0)}')\n",
    "    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
]

# Cell 2: Config
cell_config_source = [
    "# === CELL 2: HYPERPARAMETERS & CONFIGURATION ===\n",
    "# THEORETICAL NOTE (Module 3 - Optimization):\n",
    "# - Learning Rate (LR): The step size in Gradient Descent. Too high = diverge, Too low = slow.\n",
    "# - Batch Size: Number of samples per gradient update. Trade-off between noise and speed.\n",
    "# - Gradient Accumulation: Simulates a larger batch size by summing gradients over steps.\n",
    "\n",
    "CONFIG = {\n",
    "    # --- ARCHITECTURE PARAMETERS ---\n",
    "    'encoder': 'roberta-base',  # The Pre-trained Backbone (Knowledge Base)\n",
    "    'hidden_dim': 768,          # Standard BERT/RoBERTa hidden size\n",
    "    'n_heads': 8,               # Number of 'parallel eyes' in Task-Specific MHA (Module 2)\n",
    "    'num_emotion_classes': 7,   # Anger, Disgust, Fear, Joy, Sadness, Surprise, Neutral\n",
    "    'max_length': 128,          # Max tokens per sentence (Trade-off: Coverage vs Memory)\n",
    "    'dropout': 0.3,             # PROD VALUE (Log aligned). Prevents Overfitting by zeroing neurons (Module 3).\n",
    "    \n",
    "    # --- TRAINING DYNAMICS ---\n",
    "    'batch_size': 16,           # Physical batch size (fits in VRAM)\n",
    "    'gradient_accumulation': 4, # 16 * 4 = 64 Effective Batch Size (Stable Gradient Estimation)\n",
    "    'epochs': 15,               # Total training passes. (Prod value aligned with logs)\n",
    "    \n",
    "    # --- OPTIMIZATION (Module 1) ---\n",
    "    'lr_encoder': 1e-5,         # PROD VALUE. Low LR for Backbone to preserve pre-training.\n",
    "    'lr_heads': 5e-5,           # PROD VALUE. Higher LR for Heads (they are random, need fast learning).\n",
    "    'weight_decay': 0.01,       # L2 Regularization factor (penalizes large weights).\n",
    "    'max_grad_norm': 1.0,       # Gradient Clipping (Prevents Exploding Gradients).\n",
    "    'warmup_ratio': 0.1,        # 10% of steps used to warm-up LR (Stability).\n",
    "    \n",
    "    # --- ADVANCED REGULARIZATION (Module 3) ---\n",
    "    'focal_gamma': 2.0,         # Focusing parameter for Focal Loss (Focus on Hard Examples).\n",
    "    'label_smoothing': 0.1,     # Softens target labels (e.g. 0/1 -> 0.1/0.9). Improves generalization.\n",
    "    'patience': 5,              # Early Stopping counter (stops if no improvement for 5 epochs).\n",
    "    'freezing_epochs': 1,       # First epoch: Backbone frozen. Only heads train.\n",
    "}\n",
    "\n",
    "# Dataset Path (Kaggle Environment)\n",
    "DATA_DIR = '/kaggle/input/aura-v10-data'\n",
    "EMO_COLS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']\n",
    "\n",
    "print('📋 AURA V10 Configuration:')\n",
    "for k, v in CONFIG.items():\n",
    "    print(f'   {k}: {v}')"
]

# Cell 3: Viz Functions (Keeping original but adding header comment)
cell_viz_source = [
    "# === CELL 3: VISUALIZATION TOOLS ===\n",
    "# Helper functions to generate Confusion Matrices and Loss Curves.\n",
    "# Based on patterns NB10 and NB11 from the course labs.\n",
    "\n"
] + nb['cells'][3]['source']

# Cell 4: MHA Module
cell_mha_source = [
    "# === CELL 4: THE INNOVATION - TASK SPECIFIC ATTENTION ===\n",
    "# THEORETICAL NOTE (Module 2 - Attention):\n",
    "# Standard Multi-Task Learning shares the whole encoder.\n",
    "# AURA introduces 'Feature Disentanglement': each task learns to attend to different tokens.\n",
    "# - Toxicity might look at 'idiot'.\n",
    "# - Reporting might look at 'said that'.\n",
    "# This implements the 'Redundancy Principle' (using multiple heads to capture diverse features).\n",
    "\n",
    "class TaskSpecificMHA(nn.Module):\n",
    "    \"\"\"Multi-Head Self-Attention per task (Module 2: Redundancy Principle).\n",
    "    \n",
    "    Each task gets its own attention mechanism to learn WHERE to look.\n",
    "    \"\"\"\n",
    "    def __init__(self, hidden_dim, n_heads, dropout=0.1):\n",
    "        super().__init__()\n",
    "        # The Core Mechanism: Query, Key, Value\n",
    "        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V\n",
    "        self.mha = nn.MultiheadAttention(\n",
    "            embed_dim=hidden_dim, \n",
    "            num_heads=n_heads, \n",
    "            batch_first=True, \n",
    "            dropout=dropout\n",
    "        )\n",
    "        # LayerNorm & Residual Connection are crucial for Deep Learning stability (Transformers)\n",
    "        self.layernorm = nn.LayerNorm(hidden_dim)\n",
    "        self.dropout = nn.Dropout(dropout)\n",
    "        \n",
    "    def forward(self, hidden_states, attention_mask):\n",
    "        # key_padding_mask: Tells Attention to ignore PAD tokens (True = Ignore)\n",
    "        key_padding_mask = (attention_mask == 0)\n",
    "        \n",
    "        # Self-Attention: Q=K=V=hidden_states\n",
    "        # We let the model query itself to find relationships between tokens.\n",
    "        attn_output, attn_weights = self.mha(\n",
    "            query=hidden_states, \n",
    "            key=hidden_states, \n",
    "            value=hidden_states,\n",
    "            key_padding_mask=key_padding_mask\n",
    "        )\n",
    "        # Add & Norm (Residual Connection + Normalization)\n",
    "        # x = Norm(x + Dropout(Attention(x)))\n",
    "        output = self.layernorm(hidden_states + self.dropout(attn_output))\n",
    "        return output, attn_weights\n",
    "\n",
    "print('🧠 TaskSpecificMHA module defined.')"
]

# Cell 5: Model
cell_model_source = [
    "# === CELL 5: THE AURA ARCHITECTURE ===\n",
    "\n",
    "class AURA_V10(nn.Module):\n",
    "    \"\"\"AURA V10: RoBERTa + 4 Parallel Task-Specific MHSA Blocks.\"\"\"\n",
    "    \n",
    "    def __init__(self, config):\n",
    "        super().__init__()\n",
    "        # 1. The Backbone: Pre-trained RoBERTa (Frozen initially)\n",
    "        self.roberta = RobertaModel.from_pretrained(config['encoder'])\n",
    "        hidden = config['hidden_dim']\n",
    "        \n",
    "        # 2. Parallel Processing (The 4 Investigators)\n",
    "        # Instead of 1 head doing everything, we split the processing.\n",
    "        self.tox_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])\n",
    "        self.emo_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])\n",
    "        self.sent_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])\n",
    "        self.report_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])\n",
    "        \n",
    "        self.dropout = nn.Dropout(config['dropout'])\n",
    "        \n",
    "        # 3. Classification Heads (The deciders)\n",
    "        # Linear projections from Hidden Dimension -> Number of Classes\n",
    "        self.toxicity_head = nn.Linear(hidden, 2)\n",
    "        self.emotion_head = nn.Linear(hidden, config['num_emotion_classes'])\n",
    "        self.sentiment_head = nn.Linear(hidden, 2)\n",
    "        self.reporting_head = nn.Linear(hidden, 1) # Binary regression output\n",
    "        \n",
    "        # --- BIAS INITIALIZATION (Module 3 - Imbalance) ---\n",
    "        # Since Toxicity is rare (~5%), initializing bias to -0.5 lowers the starting probability.\n",
    "        # This prevents high initial loss and helps convergence.\n",
    "        with torch.no_grad():\n",
    "            self.toxicity_head.bias[0] = 0.5   # Non-Toxic\n",
    "            self.toxicity_head.bias[1] = -0.5  # Toxic\n",
    "\n",
    "    def _mean_pool(self, seq, mask):\n",
    "        \"\"\"Average Pooling Strategy.\n",
    "        Instead of using just the [CLS] token, we average all valid token embeddings.\n",
    "        This often captures more context for short texts like tweets.\n",
    "        \"\"\"\n",
    "        mask_exp = mask.unsqueeze(-1).expand(seq.size()).float()\n",
    "        return (seq * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "        # 1. Extract contextual embeddings from RoBERTa\n",
    "        shared = self.roberta(input_ids, attention_mask).last_hidden_state\n",
    "        \n",
    "        # 2. Parallel Attention Pass\n",
    "        # Each MHA block extracts task-specific features from the shared embeddings.\n",
    "        tox_seq, _ = self.tox_mha(shared, attention_mask)\n",
    "        emo_seq, _ = self.emo_mha(shared, attention_mask)\n",
    "        sent_seq, _ = self.sent_mha(shared, attention_mask)\n",
    "        rep_seq, _ = self.report_mha(shared, attention_mask)\n",
    "        \n",
    "        # 3. Pooling & Prediction\n",
    "        return {\n",
    "            'toxicity': self.toxicity_head(self.dropout(self._mean_pool(tox_seq, attention_mask))),\n",
    "            'emotion': self.emotion_head(self.dropout(self._mean_pool(emo_seq, attention_mask))),\n",
    "            'sentiment': self.sentiment_head(self.dropout(self._mean_pool(sent_seq, attention_mask))),\n",
    "            'reporting': self.reporting_head(self.dropout(self._mean_pool(rep_seq, attention_mask))).squeeze(-1)\n",
    "        }\n",
    "\n",
    "print('🦅 AURA_V10 model defined.')"
]

# Cell 6: Loss
cell_loss_source = [
    "# === CELL 6: LOSS FUNCTIONS (The Mathematical Core) ===\n",
    "# THEORETICAL NOTE (Module 3 - Advanced Learning):\n",
    "# 1. Focal Loss: Reduces loss for well-classified examples, forces focus on 'hard' misclassifications.\n",
    "# 2. Kendall Uncertainty Loss: Bayesian formulation for Multi-Task Learning.\n",
    "#    Learnable weights sigma (variance). High sigma = High Uncertainty = Lower Weight.\n",
    "\n",
    "def focal_loss(logits, targets, gamma=2.0, weight=None, smoothing=0.0):\n",
    "    \"\"\"Focal Loss Implementation.\n",
    "    Formula: FL(p_t) = -(1 - p_t)^gamma * log(p_t)\n",
    "    When p_t is high (easy example), (1-p_t) -> 0, effectively silencing the loss.\n",
    "    \"\"\"\n",
    "    ce = F.cross_entropy(logits, targets, weight=weight, reduction='none', label_smoothing=smoothing)\n",
    "    pt = torch.exp(-ce)\n",
    "    return ((1 - pt) ** gamma * ce).mean()\n",
    "\n",
    "class UncertaintyLoss(nn.Module):\n",
    "    \"\"\"Kendall et al. (2018) Implementation with Stability Fixes.\"\"\"\n",
    "    def __init__(self, n_tasks=4):\n",
    "        super().__init__()\n",
    "        # The parameters 'log_vars' are LEARNED during training via Backprop.\n",
    "        self.log_vars = nn.Parameter(torch.zeros(n_tasks))\n",
    "    \n",
    "    def forward(self, losses, mask=None):\n",
    "        total = 0\n",
    "        if mask is None:\n",
    "            mask = [1.0] * len(losses)\n",
    "            \n",
    "        for i, loss in enumerate(losses):\n",
    "            # SoftPlus ensures precision is always positive (Numerical Stability)\n",
    "            # precision = 1 / (2 * sigma^2)\n",
    "            precision = 1.0 / (F.softplus(self.log_vars[i]) + 1e-8)\n",
    "            \n",
    "            # The Kendall Formula: Loss * Precision + Log(Sigma)\n",
    "            # The model balances minimizing Loss with minimizing the Regularization term.\n",
    "            term = precision * loss + F.softplus(self.log_vars[i]) * 0.5\n",
    "            \n",
    "            # MASKING: Crucial fix for V10.2.\n",
    "            # If a task is missing in the batch (mask=0), we must zero out its contribution entirely\n",
    "            # to prevent 'Phantom Gradients' from updating that task's uncertainty.\n",
    "            total += term * mask[i]\n",
    "            \n",
    "        return total\n",
    "    \n",
    "    def get_weights(self):\n",
    "        # Helper to visualize weights: Returns 1 / sigma^2\n",
    "        return (1.0 / (F.softplus(self.log_vars) + 1e-8)).detach().cpu().numpy()\n",
    "\n",
    "print('⚖️ Loss functions defined (Focal + Kendall V10.2 Fixed).')"
]

# Cell 7: Dataset
cell_dataset_source = [
    "# === CELL 7: DATASET HANDLING ===\n",
    "# PyTorch Dataset classes handling tokenization and label format.\n",
    "# Note: We use 'task' ID to tell the Collate function how to handle the sample.\n",
    "\n",
    "class BaseDataset(Dataset):\n",
    "    def __init__(self, path, tokenizer, max_len):\n",
    "        self.df = pd.read_csv(path)\n",
    "        self.tok = tokenizer\n",
    "        self.max_len = max_len\n",
    "        \n",
    "    def __len__(self): \n",
    "        return len(self.df)\n",
    "    \n",
    "    def encode(self, text):\n",
    "        # RoBERTa Tokenization: handles standard text -> Input IDs + Attention Mask\n",
    "        return self.tok(\n",
    "            str(text), max_length=self.max_len, \n",
    "            padding='max_length', truncation=True, return_tensors='pt'\n",
    "        )\n",
    "\n",
    "# ... (Keeping Dataset logic compact, annotating Collate Fn) ...\n",
    "\n"
] + nb['cells'][7]['source'][8:] # Slicing to append original classes for brevity, focusing on Collate

# Overwrite cells in standard NB structure
nb['cells'][3]['source'] = cell_imports_source
nb['cells'][4]['source'] = cell_config_source
nb['cells'][5]['source'] = cell_viz_source
nb['cells'][6]['source'] = cell_mha_source
nb['cells'][7]['source'] = cell_model_source
nb['cells'][8]['source'] = cell_loss_source
nb['cells'][9]['source'] = cell_dataset_source

# Save
with open(OUTPUT_NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook generated successfully.")
