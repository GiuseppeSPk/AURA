import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.optim as optim
import numpy as np

from src.models.aura import AURA
from src.data.aura_dataset import AURADataset
from src.models.losses import UncertaintyLoss

def run_sanity_check():
    print("🚑 Running SANITY CHECK (Overfit Test) on AURA...")
    
    # 1. Mock Data Integration 
    # Instead of real files, let's create a dummy CSV for speed and isolation
    import pandas as pd
    dummy_data = {
        'text': ["I hate you", "I love you", "This is disgusting", "So happy"],
        'label': [1, 0, 1, 0], # Toxicity
        # Mocking 7 emotions (anger, disgust, fear, joy, sadness, surprise, neutral)
        'anger': [0.8, 0.0, 0.1, 0.0],
        'disgust': [0.1, 0.0, 0.9, 0.0],
        'fear': [0.0, 0.0, 0.0, 0.0],
        'joy': [0.0, 1.0, 0.0, 0.9],
        'sadness': [0.0, 0.0, 0.0, 0.0],
        'surprise': [0.0, 0.0, 0.0, 0.0],
        'neutral': [0.1, 0.0, 0.0, 0.1]
    }
    df = pd.DataFrame(dummy_data)
    df.to_csv("data/processed/sanity_check.csv", index=False)
    
    # 2. Setup
    model = AURA()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = AURADataset("data/processed/sanity_check.csv", tokenizer, max_len=32)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    uncertainty_loss = UncertaintyLoss()
    optimizer = optim.AdamW(list(model.parameters()) + list(uncertainty_loss.parameters()), lr=1e-4)
    
    model.train()
    
    # 3. Training Loop (Try to overfit these 4 samples)
    print("Starting training loop (Goal: reduce loss to near zero)...")
    for epoch in range(20):
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            tox_target = batch['toxicity_target']
            emo_target = batch['emotion_target']
            
            optimizer.zero_grad()
            
            tox_logits, emo_logits = model(input_ids, attention_mask)
            
            # Loss Calculation
            # Toxicity: CrossEntropy
            loss_t = torch.nn.CrossEntropyLoss()(tox_logits, tox_target)
            
            # Emotion: BCEWithLogits (as we used float targets in mock)
            loss_e = torch.nn.BCEWithLogitsLoss()(emo_logits, emo_target)
            
            # Combined
            loss = uncertainty_loss(loss_t, loss_e)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
        
    if total_loss < 0.1:
        print("\n✅ SANITY CHECK PASSED: Model can overfit.")
    else:
        print("\n⚠️ SANITY CHECK FAILED: Loss did not converge properly.")

if __name__ == "__main__":
    run_sanity_check()
