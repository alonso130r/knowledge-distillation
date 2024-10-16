import numpy as np
import os
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

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

def load_attention_weights(model_path):
    attention = []
    layers = sorted([d for d in os.listdir(model_path) if d.startswith('layer')], key=lambda x: int(x.replace('layer', '')))
    for layer in layers:
        i = extract_number_at_end(layer)
        layer_path = os.path.join(model_path, layer, f'layer_{i}_raw_attention')
        heads = sorted([f for f in os.listdir(layer_path) if f.endswith('.npy')], key=lambda x: int(x.split('_')[-1].replace('.npy', '')))
        layer_attention = []
        for head in heads:
            attention_weights = np.load(os.path.join(layer_path, head))  # Shape: [num_tokens, num_tokens]
            layer_attention.append(attention_weights)
        attention.append(layer_attention)  # List of heads per layer
    return attention  # List of layers, each containing a list of heads

models = ['raw-AI/', 'raw-base/', 'raw-finetune/', 'raw-teacher/', 'raw-reverse/']
all_models_attention = [load_attention_weights(model_path) for model_path in models]

def compute_entropy(attention_matrix):
    # Sum over the last axis to get probabilities
    attention_probs = attention_matrix / attention_matrix.sum(axis=-1, keepdims=True)
    # Compute entropy for each token's attention distribution
    entropies = entropy(attention_probs, axis=-1)
    # Average entropy over all tokens
    avg_entropy = np.mean(entropies)
    return avg_entropy

def compute_self_attention(attention_matrix):
    num_tokens = attention_matrix.shape[0]
    # Assuming attention_matrix is [num_tokens, num_tokens]
    self_attention = np.diag(attention_matrix)
    # Normalize by row sums to get probabilities
    attention_probs = attention_matrix / attention_matrix.sum(axis=-1, keepdims=True)
    self_attention_probs = self_attention / attention_matrix.sum(axis=-1)
    avg_self_attention = np.mean(self_attention_probs)
    return avg_self_attention

def compute_sparsity(attention_matrix, threshold=0.1):
    # Normalize attention
    attention_probs = attention_matrix / attention_matrix.sum(axis=-1, keepdims=True)
    # Binarize based on threshold
    sparse_attention = attention_probs > threshold
    # Calculate sparsity as the proportion of non-zero elements
    sparsity = np.mean(sparse_attention)
    return sparsity

def compute_attention_similarity(attention_matrix1, attention_matrix2):
    # Flatten the attention matrices
    flat1 = attention_matrix1.flatten()
    flat2 = attention_matrix2.flatten()
    # Normalize
    flat1 /= flat1.sum()
    flat2 /= flat2.sum()
    # Compute cosine similarity
    similarity = 1 - cosine(flat1, flat2)
    return similarity

def aggregate_metrics(model_attention):
    num_layers = len(model_attention)
    num_heads = len(model_attention[0])
    
    # Initialize dictionaries to store metrics
    entropy_per_layer = []
    self_attention_per_layer = []
    sparsity_per_layer = []
    similarity_per_layer = []
    
    for layer_idx in range(num_layers):
        entropy_heads = []
        self_attention_heads = []
        sparsity_heads = []
        # For similarity, compare each head with the previous layer's same head
        similarity_heads = []
        for head_idx in range(num_heads):
            attention = model_attention[layer_idx][head_idx]
            entropy_heads.append(compute_entropy(attention))
            self_attention_heads.append(compute_self_attention(attention))
            sparsity_heads.append(compute_sparsity(attention))
            
            if layer_idx > 0:
                prev_attention = model_attention[layer_idx-1][head_idx]
                sim = compute_attention_similarity(prev_attention, attention)
                similarity_heads.append(sim)
        
        # Aggregate metrics for the current layer
        entropy_per_layer.append(np.mean(entropy_heads))
        self_attention_per_layer.append(np.mean(self_attention_heads))
        sparsity_per_layer.append(np.mean(sparsity_heads))
        if layer_idx > 0:
            similarity_per_layer.append(np.mean(similarity_heads))
    
    return {
        'entropy': entropy_per_layer,
        'self_attention': self_attention_per_layer,
        'sparsity': sparsity_per_layer,
        'similarity': similarity_per_layer
    }



models_metrics = [aggregate_metrics(model_attention) for model_attention in all_models_attention]

model_names = ['Ground Truth', 'No prompting KD', 'Fine-tuned (No KD)', 'Teacher', 'Confidence']

metrics_names = ['entropy', 'self_attention', 'sparsity', 'similarity']
titles = ['Attention Entropy', 'Self-Attention Focus', 'Attention Sparsity', 'Attention Similarity']

for metric, title in zip(metrics_names, titles):
    plt.figure(figsize=(12, 6))
    for i, metrics in enumerate(models_metrics):
        plt.plot(metrics[metric], label=f'{model_names[i]}')
    plt.xlabel('Layer')
    plt.ylabel(title)
    plt.title(f'{title} Across Layers for All Models')
    handles,labels = plt.gca().get_legend_handles_labels()
    order = [2,1,4,3,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    # plt.show()
    plt.savefig(f'{metric}.png')
