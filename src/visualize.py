import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_attention(attention_matrix, driver_names, race_name, year, output_dir="docs/img"):
    """
    Plots the attention weights as a heatmap.
    
    attention_matrix: numpy array of shape (seq_len, seq_len)
    driver_names: list of string names for drivers (length seq_len)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # Only plot the non-padded drivers (where driver_names is not '<UNK>' or empty)
    # We assume padding drivers are filtered out before calling this function, or we do it here.
    valid_idx = [i for i, name in enumerate(driver_names) if name != '<UNK>' and name.strip() != '']
    
    if len(valid_idx) == 0:
        return
        
    attention_valid = attention_matrix[np.ix_(valid_idx, valid_idx)]
    valid_names = [driver_names[i] for i in valid_idx]
    
    sns.heatmap(attention_valid, xticklabels=valid_names, yticklabels=valid_names, 
                cmap="viridis", annot=False, fmt=".2f")
                
    plt.title(f"Transformer Attention Map: {race_name} {year}")
    plt.xlabel("Key Driver (Attended To)")
    plt.ylabel("Query Driver (Attending From)")
    
    # Rotate x labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    filename = f"{year}_{race_name.replace(' ', '_')}_attention.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    return os.path.join(output_dir, filename)
