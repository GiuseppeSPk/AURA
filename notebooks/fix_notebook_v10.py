import json
import os

NOTEBOOK_PATH = r'C:\Users\spicc\Desktop\Multimodal\AURA\notebooks\AURA_V10_Final_Fixed.ipynb'
BACKUP_PATH = NOTEBOOK_PATH + '.bak'

# 1. Backup
os.replace(NOTEBOOK_PATH, BACKUP_PATH)

# 2. Correct Code for Cell 11
CORRECT_SOURCE = [
    "# Cell 11: Training Functions\n",
    "def train_epoch(epoch):\n",
    "    model.train()\n",
    "    \n",
    "    # Progressive Freezing (NB10: Overfitting)\n",
    "    if epoch <= CONFIG['freezing_epochs']:\n",
    "        print(f'❄️ Epoch {epoch}: RoBERTa FROZEN')\n",
    "        for p in model.roberta.parameters(): \n",
    "             p.requires_grad = False\n",
    "    else:\n",
    "        print(f'🔥 Epoch {epoch}: RoBERTa UNFROZEN')\n",
    "        for p in model.roberta.parameters(): \n",
    "             p.requires_grad = True\n",
    "    \n",
    "    total_loss = 0\n",
    "    optimizer.zero_grad()\n",
    "    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')\n",
    "    \n",
    "    for step, batch in enumerate(pbar):\n",
    "        ids = batch['ids'].to(device)\n",
    "        mask = batch['mask'].to(device)\n",
    "        tasks = batch['tasks']\n",
    "        \n",
    "        # Forward pass\n",
    "        out = model(ids, mask)\n",
    "        \n",
    "        # Compute per-task losses\n",
    "        losses = []\n",
    "        \n",
    "        # Toxicity\n",
    "        if batch['tox'] is not None and (tasks == 0).sum() > 0:\n",
    "            losses.append(focal_loss(\n",
    "                out['toxicity'][tasks == 0], batch['tox'].to(device), \n",
    "                weight=tox_weights, smoothing=CONFIG['label_smoothing']\n",
    "            ))\n",
    "        else: \n",
    "            losses.append(torch.tensor(0., device=device, requires_grad=False))\n",
    "            \n",
    "        # Emotion (Multilabel BCE)\n",
    "        if batch['emo'] is not None and (tasks == 1).sum() > 0:\n",
    "            losses.append(F.binary_cross_entropy_with_logits(\n",
    "                out['emotion'][tasks == 1], batch['emo'].to(device)\n",
    "            ))\n",
    "        else: \n",
    "            losses.append(torch.tensor(0., device=device, requires_grad=False))\n",
    "            \n",
    "        # Sentiment\n",
    "        if batch['sent'] is not None and (tasks == 2).sum() > 0:\n",
    "            losses.append(focal_loss(\n",
    "                out['sentiment'][tasks == 2], batch['sent'].to(device), \n",
    "                smoothing=CONFIG['label_smoothing']\n",
    "            ))\n",
    "        else: \n",
    "            losses.append(torch.tensor(0., device=device, requires_grad=False))\n",
    "            \n",
    "        # Reporting\n",
    "        if batch['rep'] is not None and (tasks == 3).sum() > 0:\n",
    "            # Use BCE with logits on float target\n",
    "            losses.append(F.binary_cross_entropy_with_logits(\n",
    "                out['reporting'][tasks == 3], batch['rep'].float().to(device)\n",
    "            ))\n",
    "        else: \n",
    "            losses.append(torch.tensor(0., device=device, requires_grad=False))\n",
    "            \n",
    "        # Check for empty batch\n",
    "        if all((tasks == i).sum() == 0 for i in range(4)):\n",
    "            print(f\"⚠️ Warning: Empty batch at step {step}, skipping\")\n",
    "            optimizer.zero_grad()\n",
    "            continue\n",
    "\n",
    "        # Kendall weighted loss\n",
    "        loss = loss_fn(losses) / CONFIG['gradient_accumulation']\n",
    "        \n",
    "        # NaN/Inf safety check\n",
    "        if torch.isnan(loss) or torch.isinf(loss):\n",
    "            print(f\"⚠️ Warning: Invalid loss {loss.item():.4f} at step {step}, skipping batch\")\n",
    "            optimizer.zero_grad()\n",
    "            continue\n",
    "\n",
    "        # Backward pass\n",
    "        loss.backward()\n",
    "        \n",
    "        # Gradient Accumulation\n",
    "        if (step + 1) % CONFIG['gradient_accumulation'] == 0:\n",
    "            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])\n",
    "            optimizer.step()\n",
    "            scheduler.step()\n",
    "            optimizer.zero_grad()\n",
    "            \n",
    "        total_loss += loss.item() * CONFIG['gradient_accumulation']\n",
    "        pbar.set_postfix({'loss': f'{loss.item() * CONFIG[\"gradient_accumulation\"]:.3f}'})\n",
    "        \n",
    "    return total_loss / len(train_loader)\n",
    "\n",
    "@torch.no_grad()\n",
    "def evaluate():\n",
    "    model.eval()\n",
    "    preds, trues = [], []\n",
    "    for batch in val_loader:\n",
    "        out = model(batch['ids'].to(device), batch['mask'].to(device))\n",
    "        preds.extend(out['toxicity'].argmax(1).cpu().numpy())\n",
    "        trues.extend(batch['tox'].numpy())\n",
    "    return f1_score(trues, preds, average='macro', zero_division=0)\n",
    "\n",
    "print('🎯 Training functions defined.')"
]

# 3. Process the file
try:
    with open(BACKUP_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Identify Cell 11
    # We look for a cell that starts with "# Cell 11: Training Functions" or has the corrupted string
    fixed = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            content = "".join(cell['source'])
            if "Cell 11: Training Functions" in content or "FIX: Check for empty batch" in content:
                cell['source'] = CORRECT_SOURCE
                fixed = True
                print(f"✅ Found and fixed Cell {i}")
                break
    
    if not fixed:
        print("❌ Could not find the corrupted cell.")
    else:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"🚀 Fixed notebook saved to {NOTEBOOK_PATH}")

except Exception as e:
    print(f"❌ Error during fixing: {e}")
    # Restore from backup if failed
    if os.path.exists(BACKUP_PATH) and not os.path.exists(NOTEBOOK_PATH):
        os.replace(BACKUP_PATH, NOTEBOOK_PATH)
