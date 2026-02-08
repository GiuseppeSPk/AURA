import json

# Load notebook
with open('notebooks/AURA_V8_Colab.ipynb', 'r') as f:
    nb = json.load(f)

# 1. Update TRAINING CELL to save history
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '# --- MAIN LOOP ---' in source:
            new_source = source.replace(
                'best_f1 = 0\nprint(\'STARTING V8 TRAINING\')',
                'best_f1 = 0\nhistory = {\'train_loss\': [], \'train_f1\': [], \'val_f1\': []}\nprint(\'STARTING V8 TRAINING\')'
            )
            # Add append lines inside the loop
            new_source = new_source.replace(
                'print(f\'Epoch {epoch}: Train Loss={loss:.4f}, Train F1={train_f1:.4f}, Val F1={val_f1:.4f}\')',
                'print(f\'Epoch {epoch}: Train Loss={loss:.4f}, Train F1={train_f1:.4f}, Val F1={val_f1:.4f}\')\n    history[\'train_loss\'].append(loss)\n    history[\'train_f1\'].append(train_f1)\n    history[\'val_f1\'].append(val_f1)'
            )
            
            # Save history to file at the end
            new_source += "\n\n# Save history\nimport pickle\nwith open('history_v8.pkl', 'wb') as f:\n    pickle.dump(history, f)"
            
            cell['source'] = [new_source]
            print(f'Fixed Training Loop in cell {i}: Added history tracking')
            break

# 2. Add PLOTTING CELL at the end
plot_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 8. Visualization\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "def plot_training_history(history):\n",
        "    epochs = range(1, len(history['train_loss']) + 1)\n",
        "    plt.figure(figsize=(15, 5))\n",
        "    \n",
        "    # Plot Loss\n",
        "    plt.subplot(1, 2, 1)\n",
        "    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')\n",
        "    plt.title('Training Loss')\n",
        "    plt.xlabel('Epochs')\n",
        "    plt.ylabel('Loss')\n",
        "    plt.legend()\n",
        "    plt.grid(True)\n",
        "    \n",
        "    # Plot F1\n",
        "    plt.subplot(1, 2, 2)\n",
        "    plt.plot(epochs, history['train_f1'], 'b-o', label='Train F1')\n",
        "    plt.plot(epochs, history['val_f1'], 'r-s', label='Validation F1')\n",
        "    plt.title('F1 Score')\n",
        "    plt.xlabel('Epochs')\n",
        "    plt.ylabel('F1')\n",
        "    plt.legend()\n",
        "    plt.grid(True)\n",
        "    \n",
        "    plt.tight_layout()\n",
        "    plt.savefig('aura_v8_training_curves.png')\n",
        "    plt.show()\n",
        "\n",
        "if 'history' in locals():\n",
        "    plot_training_history(history)\n",
        "elif os.path.exists('history_v8.pkl'):\n",
        "    with open('history_v8.pkl', 'rb') as f:\n",
        "        history = pickle.load(f)\n",
        "    plot_training_history(history)\n",
        "else:\n",
        "    print('No history found. Training must complete first.')"
    ]
}

nb['cells'].append(plot_cell)
print('Added Plotting Cell')

with open('notebooks/AURA_V8_Colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=4)

print('NOTEBOOK UPDATED WITH VISUALIZATION')
