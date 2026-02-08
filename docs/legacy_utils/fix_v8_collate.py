import json

# Load notebook
with open('notebooks/AURA_V8_Colab.ipynb', 'r') as f:
    nb = json.load(f)

# Find the data loading cell and add custom collate
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'class TaskDataset' in source:
            # Add collate function before the class
            new_source = '''# 6. Data Loading with Custom Collate
def custom_collate(batch):
    """Handle mixed label shapes (scalar for binary, [5] for emotions)"""
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    tasks = [x['task'] for x in batch]
    
    # Pad labels to same size (max 5 for emotions)
    labels = []
    for x in batch:
        lbl = x['label']
        if lbl.dim() == 0:  # scalar -> pad to [5]
            padded = torch.zeros(5)
            padded[0] = lbl.item()
            labels.append(padded)
        else:
            labels.append(lbl)
    labels = torch.stack(labels)
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': labels, 'task': tasks}

class TaskDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len, task_type, emo_cols=None):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_type = task_type
        self.emo_cols = emo_cols or EMO_COLS
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        enc = self.tokenizer.encode_plus(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        item = {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(), 'task': self.task_type}
        if self.task_type == 'emotion':
            item['label'] = torch.tensor([float(row[c]) for c in self.emo_cols])
        else:
            item['label'] = torch.tensor(int(row['label']))
        return item

tokenizer = BertTokenizer.from_pretrained(CONFIG['encoder'])
tox_train = TaskDataset(f'{DATA_DIR}/toxicity_train.csv', tokenizer, CONFIG['max_length'], 'toxicity')
emo_train = TaskDataset(f'{DATA_DIR}/emotions_train.csv', tokenizer, CONFIG['max_length'], 'emotion')
sent_train = TaskDataset(f'{DATA_DIR}/sentiment_train.csv', tokenizer, CONFIG['max_length'], 'sentiment')
hate_train = TaskDataset(f'{DATA_DIR}/hate_train.csv', tokenizer, CONFIG['max_length'], 'hate')
tox_val = TaskDataset(f'{DATA_DIR}/toxicity_validation.csv', tokenizer, CONFIG['max_length'], 'toxicity')

train_set = ConcatDataset([tox_train, emo_train, sent_train, hate_train])
train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(tox_val, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=custom_collate)

print(f'Total Samples: {len(train_set)}')
print(f'Train Loader Batches: {len(train_loader)}')
'''
            cell['source'] = [new_source]
            print(f'Fixed cell {i}')
            break

# Also fix training to extract label correctly for binary tasks
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def train_epoch' in source:
            # Update to use label[:,0] for binary tasks
            new_train = '''# 7. Training
def train_epoch(model, loader, optimizer, scheduler, config):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc='Training')
    optimizer.zero_grad()
    tox_preds, tox_labels = [], []
    
    for step, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        tasks = batch['task']
        
        outputs = model(input_ids, attention_mask)
        loss = torch.tensor(0.0, device=device)
        
        for task in ['toxicity', 'sentiment', 'hate']:
            mask = [t == task for t in tasks]
            if sum(mask) > 0:
                task_logits = outputs[task][mask]
                task_labels = labels[mask][:, 0].long()  # Extract first element
                loss += focal_loss_with_uncertainty(task_logits, outputs['log_vars'][task], task_labels, config['focal_gamma'], config['mc_samples'])
                if task == 'toxicity':
                    tox_preds.extend(torch.argmax(task_logits, dim=1).cpu().numpy())
                    tox_labels.extend(task_labels.cpu().numpy())
        
        emo_mask = [t == 'emotion' for t in tasks]
        if sum(emo_mask) > 0:
            loss += mc_bce_loss(outputs['emotion'][emo_mask], outputs['log_vars']['emotion'], labels[emo_mask].float(), config['mc_samples'])
        
        loss = loss / config['gradient_accumulation']
        loss.backward()
        
        if (step + 1) % config['gradient_accumulation'] == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config['gradient_accumulation']
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader), f1_score(tox_labels, tox_preds, average='macro') if tox_labels else 0

@torch.no_grad()
def validate(model, loader):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc='Validating', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tox_labels = batch['label'][:, 0].long().to(device)  # Extract first element
        outputs = model(input_ids, attention_mask)
        preds.extend(torch.argmax(outputs['toxicity'], dim=1).cpu().numpy())
        labels.extend(tox_labels.cpu().numpy())
    return f1_score(labels, preds, average='macro')

# --- MAIN LOOP ---
model = AURA_MultiTask(CONFIG).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler = OneCycleLR(optimizer, max_lr=CONFIG['lr'], total_steps=len(train_loader)*CONFIG['epochs']//CONFIG['gradient_accumulation'])

best_f1 = 0
print('STARTING V8 TRAINING')
for epoch in range(1, CONFIG['epochs'] + 1):
    loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, CONFIG)
    val_f1 = validate(model, val_loader)
    print(f'Epoch {epoch}: Train Loss={loss:.4f}, Train F1={train_f1:.4f}, Val F1={val_f1:.4f}')
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'aura_v8_best.pt')
        print('  NEW BEST!')
print(f'COMPLETE. Best: {best_f1:.4f}')
'''
            cell['source'] = [new_train]
            print(f'Fixed training cell {i}')
            break

with open('notebooks/AURA_V8_Colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=4)

print('NOTEBOOK FIXED!')
