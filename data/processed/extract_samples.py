import pandas as pd
import re

# Keywords reflecting reporting/occurrence from Sprugnoli's theory
keywords = ['said', 'reporting', 'called', 'claims', 'told', 'quoted', 'denounced', 'accused', 'alleged', 'says', 'denying']
pattern = '|'.join(keywords)

df = pd.read_csv(r'c:\Users\spicc\Desktop\Multimodal\AURA\data\processed\olid_train.csv')

# Filter for rows that match any keyword
mask = df['text'].str.contains(pattern, case=False, na=False)
results = df[mask].head(100)

# Save to a new CSV for the partner
results.to_csv(r'c:\Users\spicc\Desktop\Multimodal\AURA\data\processed\event_annotation_task_MIRATO.csv', index=False)
print(f"Extracted {len(results)} samples.")
