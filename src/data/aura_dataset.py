import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import yaml

class AURADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, mode="train"):
        """
        Generic Dataset class for AURA.
        Can handle:
        1. Only Toxicity (OLID)
        2. Only Emotion (GoEmotions)
        3. Both (if we had a dataset with both, or for testing)
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        
        # Load config to know label columns
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
            
        self.emotion_labels = self.config['data']['emotion_labels']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        text = str(self.data.iloc[item]['text'])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        item_dict = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text 
        }

        # Handle Labels based on columns present in dataframe
        # Toxicity (usually 'label' column in tweet_eval/OLID where 1=offensive)
        if 'label' in self.data.columns:
            item_dict['toxicity_target'] = torch.tensor(self.data.iloc[item]['label'], dtype=torch.long)
        else:
            # If not present (e.g. goemotions), return dummy -1
            item_dict['toxicity_target'] = torch.tensor(-1, dtype=torch.long)

        # Emotion
        # Check if emotion columns exist
        if all(col in self.data.columns for col in self.emotion_labels):
            emotions = self.data.iloc[item][self.emotion_labels].values.astype(np.float32)
            # If multi-label (GoEmotions is multi-label), we might need BCE
            # But AURA architecture usually assumes Softmax (Single label) or Sigmoid?
            # Ekman mapping usually results in one dominant emotion or multi.
            # For simplicity in 'classification', let's stick to the dominant one or float vector.
            # Using float vector for BCEWithLogitsLoss or CrossEntropy if argmax.
            item_dict['emotion_target'] = torch.tensor(emotions, dtype=torch.float) 
        else:
            item_dict['emotion_target'] = torch.tensor([-1]*len(self.emotion_labels), dtype=torch.float)

        return item_dict
