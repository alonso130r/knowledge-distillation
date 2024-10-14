import os
import numpy as np
from collections import defaultdict

def extract_number_at_end(text):
    # Start from the end of the string and work backward to find the last digits
    number = ''
    for char in reversed(text):
        if char.isdigit():
            number = char + number
        else:
            break
    # Return the number as an integer if it has 1 or 2 digits
    return int(number) if 1 <= len(number) <= 2 else None

base_dir = ''

model_names = ['raw-AI/', 'raw-base/', 'raw-finetune/', 'raw-teacher/', 'raw-reverse/']

# Initialize a dictionary to store attention scores
# Structure: attention_scores[layer][head] = [model1_attention, model2_attention, ..., model5_attention]
attention_scores = defaultdict(lambda: defaultdict(list))

# Iterate over each model directory
for model_name in model_names:
    model_path = os.path.join(base_dir, model_name)
    print(f'\nProcessing model: {model_name}')
    
    # List layer directories
    layer_dirs = sorted([d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))])
    if not layer_dirs:
        print(f"  No layer directories found in {model_path}")
        continue
    
    for layer_dir in layer_dirs:
        # Construct path to attention directory
        # Adjust 'layer_x_raw_attention' if your naming is different
        i = extract_number_at_end(layer_dir)
        attention_dir = os.path.join(model_path, layer_dir, f"layer_{i}_raw_attention")
        if not os.path.isdir(attention_dir):
            # Try alternative naming if necessary
            print(f"  Attention directory not found: {attention_dir}")
            continue
        
        # Extract layer number
        try:
            layer_num = i
        except ValueError:
            print(f"  Skipping invalid layer directory: {layer_dir}")
            continue
        
        # List head files
        head_files = [f for f in os.listdir(attention_dir) if f.endswith('.npy')]
        if not head_files:
            print(f"    No head files found in {attention_dir}")
            continue
        
        for head_file in head_files:
            # Extract head number
            try:
                head_num = int(head_file.split('_')[-1].split('.')[0])
            except ValueError:
                print(f"    Skipping invalid head file: {head_file}")
                continue
            
            head_path = os.path.join(attention_dir, head_file)
            
            # Load attention scores
            try:
                # attention = np.load(head_path)  # Shape: (num_samples, seq_length, seq_length)
                # if attention.ndim != 3:
                #     print(f"    Unexpected attention shape in {head_path}: {attention.shape}")
                #     continue
                # Compute mean attention across samples
                # attention_mean = np.mean(attention, axis=0)  # Shape: (seq_length, seq_length)
                attention_mean = np.load(head_path) 
            except Exception as e:
                print(f"    Error loading {head_path}: {e}")
                continue
            
            # Store the mean attention
            attention_scores[layer_num][head_num].append(attention_mean)

# Now, compute the variance for each head across models
head_variances = []  # List to store (layer, head, variance)

for layer in attention_scores:
    for head in attention_scores[layer]:
        model_attentions = attention_scores[layer][head]
        if len(model_attentions) != 5:
            print(f"Layer {layer}, Head {head} has {len(model_attentions)} models. Skipping.")
            continue
        
        # Stack attention matrices: shape (5, seq_length, seq_length)
        stacked_attention = np.stack(model_attentions, axis=0)
        
        # Compute variance across the first axis (models)
        variance = np.var(stacked_attention, axis=0)  # Shape: (seq_length, seq_length)
        
        # Aggregate variance, e.g., mean variance across all elements
        mean_variance = np.mean(variance)
        
        head_variances.append((layer, head, mean_variance))

# Sort the heads by variance in descending order
head_variances_sorted = sorted(head_variances, key=lambda x: x[2], reverse=True)

# Define how many top divergent heads you want to investigate
top_N = 10  # Adjust as needed

selected_heads = head_variances_sorted[:top_N]

# Output the selected heads in the desired format
if selected_heads:
    print("\nHeads with the highest variance across models:")
    for idx, (layer, head, var) in enumerate(selected_heads, 1):
        print(f"{idx}. Head {head} at Layer {layer} is worth investigating (Variance: {var:.6f})")
else:
    print("\nNo heads found with complete data across all models.")