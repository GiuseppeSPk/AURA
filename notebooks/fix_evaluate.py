import json
import os

NOTEBOOKS = [
    r"C:\Users\spicc\Desktop\Multimodal\AURA\notebooks\AURA_V10_PROD.ipynb",
    r"C:\Users\spicc\Desktop\Multimodal\AURA\notebooks\AURA_V10_COLAB.ipynb"
]

def fix_evaluate_function(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    fixed = False
    for cell in nb['cells']:
        src = "".join(cell['source'])
        if "def evaluate(loader, task_id, task_name):" in src:
            print(f"👉 Found evaluate() in {os.path.basename(path)}")
            
            # The CORRECT evaluate function that handles shapes properly
            new_source = [
                "# Cell 11: Training & Validation Loop (FIXED v2)\n",
                "def evaluate(loader, task_id, task_name):\n",
                "    \"\"\"Evaluate model on a task-specific validation loader.\"\"\"\n",
                "    model.eval()\n",
                "    all_preds, all_targets = [], []\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        for batch in loader:\n",
                "            ids = batch['ids'].to(device)\n",
                "            mask = batch['mask'].to(device)\n",
                "            out = model(ids, mask)\n",
                "            \n",
                "            if task_id == 0:  # Toxicity (multiclass)\n",
                "                logits = out['toxicity']\n",
                "                y = batch['tox'].to(device)\n",
                "                preds = torch.argmax(logits, dim=1)\n",
                "                all_preds.extend(preds.cpu().numpy())\n",
                "                all_targets.extend(y.cpu().numpy())\n",
                "                \n",
                "            elif task_id == 3:  # Reporting (binary with sigmoid)\n",
                "                logits = out['reporting'].squeeze(-1)  # CRITICAL: squeeze (batch,1) -> (batch,)\n",
                "                y = batch['rep'].to(device)\n",
                "                preds = (torch.sigmoid(logits) > 0.5).int()\n",
                "                all_preds.extend(preds.cpu().numpy())\n",
                "                all_targets.extend(y.cpu().numpy())\n",
                "    \n",
                "    # Safety check\n",
                "    if len(all_preds) == 0:\n",
                "        print(f'   ⚠️ {task_name}: No samples to evaluate!')\n",
                "        return 0.0\n",
                "    \n",
                "    f1 = f1_score(all_targets, all_preds, average='macro')\n",
                "    print(f'   📊 {task_name} Val F1: {f1:.4f} (n={len(all_preds)})')\n",
                "    return f1\n",
                "\n",
                "def train_epoch(epoch):\n",
                "    model.train()\n",
                "    if epoch <= CONFIG['freezing_epochs']:\n",
                "        print(f'❄️ Epoch {epoch}: RoBERTa FROZEN')\n",
                "        for p in model.roberta.parameters(): p.requires_grad = False\n",
                "    else:\n",
                "        print(f'🔥 Epoch {epoch}: RoBERTa UNFROZEN')\n",
                "        for p in model.roberta.parameters(): p.requires_grad = True\n",
                "    \n",
                "    optimizer.zero_grad()\n",
                "    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', mininterval=10.0)\n",
                "    \n",
                "    for step, batch in enumerate(pbar):\n",
                "        ids = batch['ids'].to(device)\n",
                "        mask = batch['mask'].to(device)\n",
                "        tasks = batch['tasks']\n",
                "        out = model(ids, mask)\n",
                "        \n",
                "        losses = []\n",
                "        # Toxicity\n",
                "        if batch['tox'] is not None and (tasks == 0).sum() > 0:\n",
                "            losses.append(focal_loss(out['toxicity'][tasks == 0], batch['tox'].to(device), weight=tox_weights, smoothing=CONFIG['label_smoothing']))\n",
                "        else: losses.append(torch.tensor(0., device=device, requires_grad=False))\n",
                "            \n",
                "        # Emotion\n",
                "        if batch['emo'] is not None and (tasks == 1).sum() > 0:\n",
                "            losses.append(F.binary_cross_entropy_with_logits(out['emotion'][tasks == 1], batch['emo'].to(device)))\n",
                "        else: losses.append(torch.tensor(0., device=device, requires_grad=False))\n",
                "            \n",
                "        # Sentiment\n",
                "        if batch['sent'] is not None and (tasks == 2).sum() > 0:\n",
                "            losses.append(focal_loss(out['sentiment'][tasks == 2], batch['sent'].to(device), smoothing=CONFIG['label_smoothing']))\n",
                "        else: losses.append(torch.tensor(0., device=device, requires_grad=False))\n",
                "            \n",
                "        # Reporting\n",
                "        if batch['rep'] is not None and (tasks == 3).sum() > 0:\n",
                "            rep_logits = out['reporting'][tasks == 3].squeeze(-1)  # FIXED: squeeze\n",
                "            losses.append(F.binary_cross_entropy_with_logits(rep_logits, batch['rep'].float().to(device)))\n",
                "        else: losses.append(torch.tensor(0., device=device, requires_grad=False))\n",
                "\n",
                "        if all((tasks == i).sum() == 0 for i in range(4)):\n",
                "             continue\n",
                "\n",
                "        loss = loss_fn(losses) / CONFIG['gradient_accumulation']\n",
                "        if torch.isnan(loss) or torch.isinf(loss): continue\n",
                "        loss.backward()\n",
                "        \n",
                "        if (step + 1) % CONFIG['gradient_accumulation'] == 0:\n",
                "            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])\n",
                "            optimizer.step()\n",
                "            optimizer.zero_grad()\n",
                "            scheduler.step()\n",
                "\n",
                "    # End of Epoch Validation\n",
                "    print(f'\\n📝 Epoch {epoch} Validation:')\n",
                "    val_f1_tox = evaluate(val_loader_tox, 0, 'Toxicity')\n",
                "    val_f1_rep = evaluate(val_loader_rep, 3, 'Reporting')\n",
                "    \n",
                "    return {'val_f1': val_f1_tox, 'val_f1_rep': val_f1_rep}\n"
            ]
            cell['source'] = new_source
            fixed = True
            break

    if fixed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"✅ Fixed {os.path.basename(path)}")
    else:
        print(f"⚠️ Could not find evaluate() in {os.path.basename(path)}")

if __name__ == "__main__":
    for nb_path in NOTEBOOKS:
        fix_evaluate_function(nb_path)
    print("\n✅ Bug fix complete! Key changes:")
    print("   1. Removed task filtering in evaluate() - loader is already task-specific")
    print("   2. Added .squeeze(-1) to reporting logits to match label shape")
    print("   3. Added sample count to F1 output for verification")
