# -*- coding: utf-8 -*-
"""
DNAGPT Privacy-Utility Analysis - ENHANCED COMPLETE VERSION
Implementation based on Zhang et al. (2023)
Using non-overlapped k-mer tokenization as per original paper

Architecture:
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- ~100M parameters
- Non-overlapped k-mer tokenization (k=6)
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import GPT2Config, GPT2ForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
import matplotlib.ticker as mtick
import json
from itertools import product

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
train_path = 'hs3d_train_pan.csv'
test_path = 'hs3d_test_pan.csv'
output_dir = "./dnagpt-result-complete/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}attack_results/", exist_ok=True)

print("="*80)
print("DNAGPT PRIVACY-UTILITY ANALYSIS - ENHANCED COMPLETE VERSION")
print("="*80)
print("Implementation: Zhang et al. (2023)")
print("Paper: 'DNAGPT: A Generalized Pre-trained Tool for")
print("        Versatile DNA Sequence Analysis Tasks'")
print("="*80)
print("Architecture Specifications:")
print("  - Tokenization: Non-overlapped k-mers (k=6)")
print("  - Layers: 12 transformer blocks")
print("  - Hidden dimensions: 768")
print("  - Attention heads: 12")
print("  - Parameters: ~100M")
print("  - Vocabulary: 4^6 = 4,096 k-mers + 4 special tokens")
print("="*80)

# ============================================================================
# DNAGPT TOKENIZER (Non-overlapped k-mers - Zhang et al. 2023)
# ============================================================================

def create_dnagpt_tokenizer(k=6):
    """
    Create DNAGPT tokenizer using non-overlapped k-mers
    
    Following Zhang et al. (2023):
    - Non-overlapped k-mers with shift = k
    - Provides k-fold efficiency improvement
    - Vocabulary: 4^k possible k-mers
    
    Args:
        k (int): k-mer size (default: 6, as per paper)
    
    Returns:
        vocab (dict): Token to ID mapping
        k (int): k-mer size
    """
    print(f"\n{'='*80}")
    print("CREATING DNAGPT TOKENIZER")
    print(f"{'='*80}")
    print(f"k-mer size: {k}")
    print(f"Tokenization strategy: Non-overlapped (shift = {k})")
    
    nucleotides = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(nucleotides, repeat=k)]
    
    # Create vocabulary
    vocab = {}
    special_tokens = ['<PAD>', '<EOS>', '<UNK>', '<BOS>']
    
    # Add special tokens
    for idx, token in enumerate(special_tokens):
        vocab[token] = idx
    
    # Add all k-mers
    for idx, kmer in enumerate(kmers):
        vocab[kmer] = len(special_tokens) + idx
    
    # Save vocabulary
    vocab_file = f"{output_dir}dnagpt_vocab.json"
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Vocabulary created:")
    print(f"  - Total tokens: {len(vocab)}")
    print(f"  - Special tokens: {len(special_tokens)} {special_tokens}")
    print(f"  - k-mers: {len(kmers)} (4^{k})")
    print(f"  - Example k-mers: {kmers[:10]}")
    print(f"  - Vocabulary saved: {vocab_file}")
    print(f"{'='*80}")
    
    return vocab, k

def tokenize_dna_sequence(sequence, vocab, k, max_length=128):
    """
    Tokenize DNA sequence using non-overlapped k-mers
    
    Following Zhang et al. (2023) methodology:
    - Segments sequence into non-overlapping k-mers
    - For sequence of length N, produces ~N/k tokens
    - k-fold efficiency improvement over character-level
    
    Args:
        sequence (str): DNA sequence
        vocab (dict): Token to ID mapping
        k (int): k-mer size
        max_length (int): Maximum sequence length in tokens
    
    Returns:
        tokens (list): Token IDs
        attention_mask (list): Attention mask
    """
    # Clean sequence
    sequence = sequence.upper().replace(' ', '').replace('\n', '')
    
    tokens = []
    
    # Non-overlapped k-mers: shift = k (key difference from overlapped)
    for i in range(0, len(sequence), k):
        kmer = sequence[i:i+k]
        if len(kmer) == k:  # Only complete k-mers
            token_id = vocab.get(kmer, vocab['<UNK>'])
            tokens.append(token_id)
    
    # Add EOS token
    tokens.append(vocab['<EOS>'])
    
    # Pad or truncate
    if len(tokens) < max_length:
        tokens = tokens + [vocab['<PAD>']] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length-1] + [vocab['<EOS>']]
    
    # Create attention mask
    attention_mask = [1 if token != vocab['<PAD>'] else 0 for token in tokens]
    
    return tokens, attention_mask

def create_dnagpt_config(vocab_size):
    """
    Create DNAGPT configuration matching Zhang et al. (2023)
    
    Specifications from paper:
    - 12 layers
    - 768 hidden dimensions
    - 12 attention heads
    - ~100M parameters
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,      # Maximum sequence length
        n_embd=768,           # Hidden size (as per paper)
        n_layer=12,           # Number of layers (as per paper)
        n_head=12,            # Number of attention heads (as per paper)
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        use_cache=True,
    )
    return config

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and preprocess HS3D dataset"""
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['sequence'] = train_df['sequence'].astype(str)
    test_df['sequence'] = test_df['sequence'].astype(str)

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Training class distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"Testing class distribution: {test_df['label'].value_counts().to_dict()}")

    # Visualize class distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='label', data=train_df, palette='viridis')
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.countplot(x='label', data=test_df, palette='viridis')
    plt.title('Class Distribution in Test Set')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(f"{output_dir}class_distribution.png", dpi=300)
    plt.close()
    
    print(f"? Class distribution plot saved")
    print(f"{'='*80}")

    return train_df, test_df

# ============================================================================
# DATASET CREATION
# ============================================================================

def create_datasets(train_df, test_df, vocab, k=6, max_length=128):
    """
    Create tokenized datasets using DNAGPT k-mer tokenization
    
    Args:
        train_df, test_df: DataFrames with 'sequence' and 'label' columns
        vocab: Token to ID mapping
        k: k-mer size
        max_length: Maximum sequence length in tokens
    """
    print(f"\n{'='*80}")
    print("TOKENIZING SEQUENCES")
    print(f"{'='*80}")
    print(f"Tokenization: Non-overlapped {k}-mers")
    print(f"Max length: {max_length} tokens")
    
    # Tokenize training data
    train_tokens = []
    train_attention_masks = []
    for seq in train_df['sequence'].tolist():
        tokens, attention_mask = tokenize_dna_sequence(seq, vocab, k, max_length)
        train_tokens.append(tokens)
        train_attention_masks.append(attention_mask)
    
    # Tokenize test data
    test_tokens = []
    test_attention_masks = []
    for seq in test_df['sequence'].tolist():
        tokens, attention_mask = tokenize_dna_sequence(seq, vocab, k, max_length)
        test_tokens.append(tokens)
        test_attention_masks.append(attention_mask)
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_tokens,
        'attention_mask': train_attention_masks,
        'labels': train_df['label'].tolist()
    })
    
    test_dataset = Dataset.from_dict({
        'input_ids': test_tokens,
        'attention_mask': test_attention_masks,
        'labels': test_df['label'].tolist()
    })
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Visualize sequence length distribution
    seq_lengths = [len(seq) for seq in train_df['sequence']]
    kmer_lengths = [len(seq) // k for seq in train_df['sequence']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(seq_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Sequence Lengths (Nucleotides)', fontsize=14)
    ax1.set_xlabel('Sequence Length (bp)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.axvline(np.mean(seq_lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(seq_lengths):.1f}')
    ax1.legend()
    
    ax2.hist(kmer_lengths, bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_title(f'Distribution of Tokenized Lengths ({k}-mers)', fontsize=14)
    ax2.set_xlabel(f'Number of {k}-mers', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.axvline(np.mean(kmer_lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(kmer_lengths):.1f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}sequence_length_distribution.png", dpi=300)
    plt.close()
    
    print(f"Sequence statistics:")
    print(f"  - Average nucleotide length: {np.mean(seq_lengths):.1f} bp")
    print(f"  - Average token count: {np.mean(kmer_lengths):.1f} {k}-mers")
    print(f"  - Efficiency gain: {k}x (vs character-level)")
    print(f"  - Max nucleotide length: {np.max(seq_lengths)}")
    print(f"  - Min nucleotide length: {np.min(seq_lengths)}")
    print(f"? Sequence length distribution plot saved")
    print(f"{'='*80}")
    
    return train_dataset, test_dataset

# ============================================================================
# DNAGPT MODEL INITIALIZATION & FINE-TUNING
# ============================================================================

def initialize_dnagpt(vocab_size):
    """
    Initialize DNAGPT model with random weights
    
    Note: Official pre-trained weights not publicly available
    Following Zhang et al. (2023) architecture specifications
    """
    print(f"\n{'='*80}")
    print("INITIALIZING DNAGPT MODEL")
    print(f"{'='*80}")
    
    config = create_dnagpt_config(vocab_size)
    config.pad_token_id = 0  # <PAD> token ID
    
    model = GPT2ForSequenceClassification(config)
    model.config.num_labels = 2
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model architecture:")
    print(f"  - Layers: {config.n_layer}")
    print(f"  - Hidden dimensions: {config.n_embd}")
    print(f"  - Attention heads: {config.n_head}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  - Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"  - Initialization: Random (pre-trained weights not available)")
    print(f"{'='*80}")
    
    return model

def finetune_dnagpt(train_dataset, test_dataset, model):
    """Fine-tune DNAGPT on task-specific dataset"""
    print(f"\n{'='*80}")
    print("FINE-TUNING DNAGPT")
    print(f"{'='*80}")
    print("Training configuration (following Zhang et al. 2023):")
    print("  - Learning rate: 1e-4")
    print("  - Weight decay: 1e-2")
    print("  - Batch size: 16 (effective: 32 with gradient accumulation)")
    print("  - Epochs: 3")
    print("  - Optimizer: Adam")
    print(f"{'='*80}")

    training_args = TrainingArguments(
        output_dir=f"{output_dir}finetune",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=1e-2,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        try:
            auc = roc_auc_score(labels, pred.predictions[:, 1])
        except:
            auc = 0.0
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("\nStarting training...")
    trainer.train()
    
    eval_result = trainer.evaluate()
    
    print(f"\n{'='*80}")
    print("FINE-TUNING RESULTS")
    print(f"{'='*80}")
    for key, value in eval_result.items():
        if not key.startswith('eval_'):
            continue
        metric_name = key.replace('eval_', '').upper()
        print(f"  {metric_name}: {value:.4f}")
    print(f"{'='*80}")

    model.save_pretrained(f"{output_dir}finetune_dnagpt")
    print(f"? Model saved to: {output_dir}finetune_dnagpt")
    
    return model, eval_result

# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_embeddings(model, sequences, vocab, k=6, max_length=128):
    """
    Extract embeddings from DNAGPT model
    Uses last token representation (autoregressive architecture)
    """
    print(f"\n{'='*80}")
    print("EXTRACTING EMBEDDINGS")
    print(f"{'='*80}")
    print(f"Extraction method: Last token (autoregressive)")
    
    feature_extractor = model.transformer
    feature_extractor.eval()
    all_embeddings = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]

            # Tokenize batch
            batch_tokens = []
            batch_attention_masks = []
            for seq in batch_sequences:
                tokens, attention_mask = tokenize_dna_sequence(seq, vocab, k, max_length)
                batch_tokens.append(tokens)
                batch_attention_masks.append(attention_mask)
            
            # Convert to tensors
            input_ids = torch.tensor(batch_tokens, dtype=torch.long)
            attention_mask = torch.tensor(batch_attention_masks, dtype=torch.long)

            # Move to GPU if available
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                feature_extractor = feature_extractor.cuda()

            # Forward pass
            outputs = feature_extractor(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state
            
            # Get actual sequence lengths (exclude padding)
            seq_lengths = attention_mask.sum(dim=1) - 1
            
            # Extract last token embedding for each sequence
            batch_embeddings = []
            for idx, seq_len in enumerate(seq_lengths):
                last_token_embedding = last_hidden_states[idx, seq_len, :].cpu().numpy()
                batch_embeddings.append(last_token_embedding)
            
            all_embeddings.append(np.array(batch_embeddings))
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + batch_size}/{len(sequences)} sequences...")

    embeddings = np.vstack(all_embeddings)
    
    print(f"? Extraction complete")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embedding dimension: {embeddings.shape[1]} (matches hidden size: 768)")
    print(f"{'='*80}")

    return embeddings

# ============================================================================
# MLP CLASSIFIER
# ============================================================================

class MLPClassifier(nn.Module):
    """
    3-layer MLP classifier for embeddings
    Following Pan et al. methodology
    """
    def __init__(self, input_dim, hidden_dim=200, num_classes=2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.bn1(self.sigmoid(self.fc1(x)))
        x = self.bn2(self.sigmoid(self.fc2(x)))
        x = self.fc3(x)
        return x

def train_mlp(train_embeddings, train_labels, test_embeddings, test_labels, model_type):
    """Train MLP classifier on embeddings"""
    print(f"\n{'='*80}")
    print(f"TRAINING MLP CLASSIFIER - {model_type.upper()}")
    print(f"{'='*80}")

    train_data = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    test_data = TensorDataset(
        torch.tensor(test_embeddings, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    input_dim = train_embeddings.shape[1]
    model = MLPClassifier(input_dim=input_dim)

    if torch.cuda.is_available():
        model = model.cuda()
        print("? Using GPU")
    else:
        print("? Using CPU")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_losses = []
    val_accuracies = []
    all_epoch_metrics = []

    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                outputs = model(batch_x)
                probabilities = torch.softmax(outputs, dim=1)
                all_probs.extend(probabilities[:, 1].cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(accuracy)

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_probs)

        all_epoch_metrics.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        })

        print(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={accuracy:.4f}, "
              f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    torch.save(model.state_dict(), f"{output_dir}mlp_classifier_{model_type}.pt")
    print(f"? Model saved")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix - {model_type}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}confusion_matrix_{model_type}.png", dpi=300)
    plt.close()

    # Learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'o-', linewidth=2, color='crimson')
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), val_accuracies, 'o-', linewidth=2, color='green')
    plt.title('Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}learning_curves_{model_type}.png", dpi=300)
    plt.close()

    # t-SNE visualization
    sample_size = min(1000, len(test_embeddings))
    indices = np.random.choice(len(test_embeddings), sample_size, replace=False)
    sampled_embeddings = test_embeddings[indices]
    sampled_labels = np.array(test_labels)[indices]

    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(sampled_embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=sampled_labels, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Class')
    plt.title(f't-SNE Visualization - {model_type} Embeddings', fontsize=14)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}tsne_{model_type}.png", dpi=300)
    plt.close()
    
    print(f"? All visualizations saved")
    print(f"{'='*80}")

    return model, all_epoch_metrics

# ============================================================================
# COMPREHENSIVE VISUALIZATIONS
# ============================================================================

def create_comprehensive_visualizations(all_metrics, model_name="DNAGPT"):
    """Create comprehensive training visualizations"""
    print(f"\nGenerating comprehensive visualizations for {model_name}...")
    
    epochs = [m['epoch'] for m in all_metrics]
    loss = [m['loss'] for m in all_metrics]
    accuracy = [m['accuracy'] for m in all_metrics]
    precision = [m['precision'] for m in all_metrics]
    recall = [m['recall'] for m in all_metrics]
    f1 = [m['f1'] for m in all_metrics]
    auc = [m['auc'] for m in all_metrics]

    # Calculate statistics
    avg_loss = np.mean(loss)
    std_loss = np.std(loss)
    avg_accuracy = np.mean(accuracy)
    std_accuracy = np.std(accuracy)
    avg_precision = np.mean(precision)
    std_precision = np.std(precision)
    avg_recall = np.mean(recall)
    std_recall = np.std(recall)
    avg_f1 = np.mean(f1)
    std_f1 = np.std(f1)
    avg_auc = np.mean(auc)
    std_auc = np.std(auc)

    # 1. Training metrics over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracy, 'o-', linewidth=2, label='Accuracy', markersize=6)
    plt.plot(epochs, precision, 's-', linewidth=2, label='Precision', markersize=6)
    plt.plot(epochs, recall, '^-', linewidth=2, label='Recall', markersize=6)
    plt.plot(epochs, f1, 'd-', linewidth=2, label='F1 Score', markersize=6)
    plt.plot(epochs, auc, 'v-', linewidth=2, label='AUC', markersize=6)

    plt.axhline(y=avg_accuracy, color='blue', linestyle='--', alpha=0.4)
    plt.axhline(y=avg_precision, color='orange', linestyle='--', alpha=0.4)
    plt.axhline(y=avg_recall, color='green', linestyle='--', alpha=0.4)
    plt.axhline(y=avg_f1, color='red', linestyle='--', alpha=0.4)
    plt.axhline(y=avg_auc, color='purple', linestyle='--', alpha=0.4)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'{model_name} - Training Metrics by Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}training_metrics_{model_name.lower()}.png", dpi=300)
    plt.close()

    # 2. Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'o-', color='crimson', linewidth=2, markersize=6)
    plt.axhline(y=avg_loss, color='crimson', linestyle='--', alpha=0.5, 
                label=f'Avg: {avg_loss:.4f}')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss by Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}loss_curve_{model_name.lower()}.png", dpi=300)
    plt.close()

    # 3. Precision-Recall trade-off
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'o-', color='teal', linewidth=2, markersize=8)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    
    for i, epoch in enumerate(epochs):
        plt.annotate(f'E{epoch}', (recall[i], precision[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}precision_recall_{model_name.lower()}.png", dpi=300)
    plt.close()

    # 4. F1 vs Accuracy scatter
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(accuracy, f1, c=epochs, cmap='viridis', s=100, alpha=0.8)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Accuracy by Epoch', fontsize=14, fontweight='bold')
    
    for i, epoch in enumerate(epochs):
        plt.annotate(f'E{epoch}', (accuracy[i], f1[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Epoch', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}f1_accuracy_{model_name.lower()}.png", dpi=300)
    plt.close()

    # 5. Average metrics bar chart
    plt.figure(figsize=(10, 6))
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    avg_values = [avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1, avg_auc]
    std_values = [std_loss, std_accuracy, std_precision, std_recall, std_f1, std_auc]

    bars = plt.bar(metrics, avg_values, yerr=std_values, capsize=10, 
                   color=sns.color_palette("muted"))
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_values[i] + 0.02,
                f'{avg_values[i]:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.ylim(0, 1.1)
    plt.title('Average Performance Metrics Across All Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Value', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}average_metrics_{model_name.lower()}.png", dpi=300)
    plt.close()

    # 6. Metrics table
    plt.figure(figsize=(12, 5))
    plt.axis('off')

    table_data = [['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']]
    for i in range(len(epochs)):
        table_data.append([
            f'{epochs[i]}',
            f'{loss[i]:.4f}',
            f'{accuracy[i]:.4f}',
            f'{precision[i]:.4f}',
            f'{recall[i]:.4f}',
            f'{f1[i]:.4f}',
            f'{auc[i]:.4f}'
        ])
    table_data.append([
        'Average',
        f'{avg_loss:.4f} +/- {std_loss:.4f}',
        f'{avg_accuracy:.4f} +/- {std_accuracy:.4f}',
        f'{avg_precision:.4f} +/- {std_precision:.4f}',
        f'{avg_recall:.4f} +/- {std_recall:.4f}',
        f'{avg_f1:.4f} +/- {std_f1:.4f}',
        f'{avg_auc:.4f} +/- {std_auc:.4f}'
    ])

    table = plt.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
            elif i == len(table_data) - 1:
                cell.set_facecolor('#E2EFDA')
                cell.set_text_props(fontweight='bold')
            elif j == 0:
                cell.set_facecolor('#D9E1F2')

    plt.title(f'{model_name} - Performance Metrics by Epoch', fontsize=14, 
              fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}metrics_table_{model_name.lower()}.png", dpi=300)
    plt.close()
    
    print(f"? All comprehensive visualizations saved")

    return {
        "accuracy": avg_accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "auc": avg_auc
    }

# ============================================================================
# PCA VISUALIZATION
# ============================================================================

def visualize_embeddings_pca(embeddings, labels, dataset_name):
    """Create PCA visualization of embeddings"""
    print(f"\nGenerating PCA visualization for {dataset_name}...")
    
    pca = PCA(n_components=2)
    sample_size = min(2000, len(embeddings))
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sampled_embeddings = embeddings[indices]
    sampled_labels = labels[indices]

    embeddings_2d = pca.fit_transform(sampled_embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=sampled_labels, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Class')
    plt.title(f'PCA Visualization - {dataset_name}', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}pca_{dataset_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()

    print(f"  PC1 explains: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2 explains: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"  Total variance: {sum(pca.explained_variance_ratio_[:2]):.2%}")
    print(f"? PCA visualization saved")

# ============================================================================
# RECONSTRUCTION ATTACK
# ============================================================================

class ReconstructionAttackModel(nn.Module):
    """Reconstruction attack model (Pan et al. methodology)"""
    def __init__(self, embedding_dim, hidden_dim=200):
        super(ReconstructionAttackModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)  # 4 nucleotides
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.bn1(self.sigmoid(self.fc1(x)))
        x = self.bn2(self.sigmoid(self.fc2(x)))
        x = self.fc3(x)
        return x

def create_positional_embedding(position, dim):
    """Create sinusoidal positional embedding"""
    pos_embedding = torch.zeros(dim)
    for i in range(0, dim, 2):
        pos_embedding[i] = np.sin(position / (10000 ** (i / dim)))
        if i + 1 < dim:
            pos_embedding[i + 1] = np.cos(position / (10000 ** ((i + 1) / dim)))
    return pos_embedding

def run_reconstruction_attack(embeddings, sequences, model_name):
    """Run reconstruction attack on embeddings"""
    print(f"\n{'='*80}")
    print(f"RECONSTRUCTION ATTACK - {model_name.upper()}")
    print(f"{'='*80}")

    embedding_dim = embeddings.shape[1]
    seq_length = 20

    # Filter valid sequences
    valid_indices = [i for i, seq in enumerate(sequences) if len(seq) >= seq_length]
    if len(valid_indices) < len(sequences):
        print(f"? Warning: {len(sequences) - len(valid_indices)} sequences shorter than {seq_length}")
        embeddings = embeddings[valid_indices]
        sequences = [sequences[i] for i in valid_indices]

    nuc_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    int_to_nuc = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    position_accuracy = []
    confusion_matrices = []

    print(f"Training attack models for {seq_length} positions...")
    
    for pos in range(seq_length):
        # Create positional embeddings
        pos_embeddings = torch.stack([create_positional_embedding(pos, embedding_dim)
                                    for _ in range(len(embeddings))])
        
        # Create labels
        y_true = torch.tensor([nuc_to_int.get(seq[pos].upper(), 0) for seq in sequences])
        
        # Concatenate embeddings with positional encoding
        X = torch.cat([torch.tensor(embeddings, dtype=torch.float32), pos_embeddings], dim=1)

        # Split data
        dataset = TensorDataset(X, y_true)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Initialize attack model
        model = ReconstructionAttackModel(embedding_dim)
        if torch.cuda.is_available():
            model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train attack model
        for epoch in range(5):
            model.train()
            for batch_x, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())

        acc = accuracy_score(all_true, all_preds)
        position_accuracy.append(acc)

        cm = confusion_matrix(all_true, all_preds, labels=[0, 1, 2, 3])
        confusion_matrices.append(cm)
        
        print(f"  Position {pos+1:2d}/{seq_length}: Accuracy = {acc:.4f}")

    print(f" Attack training complete")
    print(f"  Average reconstruction accuracy: {np.mean(position_accuracy):.4f}")

    # Visualization 1: Position-wise accuracy
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(1, seq_length + 1)
    plt.plot(x_pos, position_accuracy, 'o-', linewidth=2, color='blue', markersize=6)
    plt.axhline(y=np.mean(position_accuracy), color='red', linestyle='--',
                linewidth=2, label=f'Average: {np.mean(position_accuracy):.3f}')
    plt.axhline(y=0.25, color='gray', linestyle=':', linewidth=2, 
                label='Random Baseline (0.25)')
    plt.title(f'Reconstruction Attack Accuracy - {model_name}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(x_pos)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/reconstruction_accuracy_{model_name}.png", dpi=300)
    plt.close()

    # Visualization 2: All confusion matrices
    plt.figure(figsize=(15, 10))
    for i, cm in enumerate(confusion_matrices):
        if i < seq_length:
            plt.subplot(5, 4, i + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['A', 'C', 'G', 'T'],
                        yticklabels=['A', 'C', 'G', 'T'])
            plt.title(f'Position {i+1}', fontsize=10)
    plt.suptitle(f'Confusion Matrices - All Positions ({model_name})', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/all_confusion_matrices_{model_name}.png", dpi=300)
    plt.close()

    # Calculate nucleotide-wise accuracy
    class_accuracy = {}
    for nuc in ['A', 'C', 'G', 'T']:
        nuc_idx = nuc_to_int[nuc]
        accuracies = []
        for cm in confusion_matrices:
            if cm[nuc_idx].sum() > 0:
                acc = cm[nuc_idx, nuc_idx] / cm[nuc_idx].sum()
                accuracies.append(acc)
        if accuracies:
            class_accuracy[nuc] = np.mean(accuracies)

    # Visualization 3: Nucleotide-wise accuracy
    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_accuracy.keys(), class_accuracy.values(), 
                   color='teal', edgecolor='black', linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.axhline(y=0.25, color='red', linestyle='--', linewidth=2, 
                label='Random Baseline (0.25)')
    plt.ylim(0, 1.0)
    plt.title(f'Nucleotide-wise Reconstruction Accuracy - {model_name}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Average Accuracy', fontsize=12)
    plt.xlabel('Nucleotide', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/nucleotide_accuracy_{model_name}.png", dpi=300)
    plt.close()
    
    print(f"? Attack visualizations saved")
    print(f"  Nucleotide-wise accuracy: {class_accuracy}")
    print(f"{'='*80}")
    
    return position_accuracy, confusion_matrices, class_accuracy

# ============================================================================
# COMPARE ATTACKS
# ============================================================================

def compare_reconstruction_attacks(pretrained_results, finetuned_results):
    """Compare reconstruction attacks between pretrained and fine-tuned"""
    print(f"\n{'='*80}")
    print("COMPARING RECONSTRUCTION ATTACKS")
    print(f"{'='*80}")
    
    pretrained_acc = pretrained_results[0]
    finetuned_acc = finetuned_results[0]

    # Visualization 1: Comparison plot
    plt.figure(figsize=(12, 7))
    x_pos = np.arange(1, len(pretrained_acc) + 1)

    plt.plot(x_pos, pretrained_acc, 'o-', linewidth=2, color='blue',
             markersize=6, label=f'Pretrained (Avg: {np.mean(pretrained_acc):.3f})')
    plt.plot(x_pos, finetuned_acc, 's-', linewidth=2, color='red',
             markersize=6, label=f'Fine-tuned (Avg: {np.mean(finetuned_acc):.3f})')

    plt.axhline(y=0.25, color='gray', linestyle=':', linewidth=2, 
                label='Random Baseline (0.25)')

    # Highlight positions where privacy improved
    for i in range(len(pretrained_acc)):
        if finetuned_acc[i] < pretrained_acc[i]:
            plt.axvspan(i+0.5, i+1.5, alpha=0.2, color='green')

    plt.title('Reconstruction Attack Comparison: Pretrained vs Fine-tuned', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(x_pos)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/reconstruction_comparison.png", dpi=300)
    plt.close()

    # Calculate privacy change
    delta_privacy = np.array(pretrained_acc) - np.array(finetuned_acc)
    avg_privacy_change = np.mean(delta_privacy)

    # Visualization 2: Privacy change
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in delta_privacy]
    plt.bar(x_pos, delta_privacy, color=colors, edgecolor='black', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=avg_privacy_change, color='blue', linestyle='--', linewidth=2,
                label=f'Average: {avg_privacy_change:.3f}')

    plt.title('Privacy Change After Fine-tuning (Positive = More Private)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Accuracy Difference (Pretrained - Finetuned)', fontsize=12)
    plt.xticks(x_pos)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/privacy_change.png", dpi=300)
    plt.close()

    # Nucleotide comparison
    pretrained_nuc = pretrained_results[2]
    finetuned_nuc = finetuned_results[2]

    plt.figure(figsize=(10, 6))
    nucleotides = ['A', 'C', 'G', 'T']
    pretrained_values = [pretrained_nuc.get(nuc, 0) for nuc in nucleotides]
    finetuned_values = [finetuned_nuc.get(nuc, 0) for nuc in nucleotides]

    x = np.arange(len(nucleotides))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pretrained_values, width, label='Pretrained',
                    edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, finetuned_values, width, label='Fine-tuned',
                    edgecolor='black', linewidth=1)

    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Nucleotide-wise Reconstruction Accuracy Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(nucleotides)
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2, 
               label='Random Baseline (0.25)')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(f"{output_dir}attack_results/nucleotide_comparison.png", dpi=300)
    plt.close()

    print(f"Comparison results:")
    print(f"  Pretrained avg accuracy: {np.mean(pretrained_acc):.4f}")
    print(f"  Fine-tuned avg accuracy: {np.mean(finetuned_acc):.4f}")
    print(f"  Privacy change (delta): {avg_privacy_change:.4f}")
    
    if avg_privacy_change > 0:
        print(f"  ? Fine-tuning IMPROVED privacy (lower attack accuracy)")
    else:
        print(f"  ? Fine-tuning REDUCED privacy (higher attack accuracy)")
    
    print(f"? Comparison visualizations saved")
    print(f"{'='*80}")
        
    return delta_privacy, pretrained_acc, finetuned_acc

# ============================================================================
# PRIVACY METRICS
# ============================================================================

def calculate_privacy_gain_log(pretrained_acc, finetuned_acc):
    """Calculate error-based privacy gain with logarithmic transformation"""
    pretrained_error = 1 - np.array(pretrained_acc)
    finetuned_error = 1 - np.array(finetuned_acc)
    
    privacy_gain = np.zeros_like(pretrained_error)
    for i in range(len(pretrained_error)):
        if pretrained_error[i] > 0 and finetuned_error[i] > 0:
            ratio = finetuned_error[i] / pretrained_error[i]
            ratio = min(ratio, 1.5)
            privacy_gain[i] = np.log(ratio)
        elif finetuned_error[i] > pretrained_error[i]:
            privacy_gain[i] = np.log(1.5)
        else:
            privacy_gain[i] = 0
    
    return privacy_gain

def calculate_pds_delta(delta_privacy, utility_delta):
    """Calculate Pareto Dominance Score based on delta privacy"""
    pds = np.zeros(len(delta_privacy))
    for i in range(len(delta_privacy)):
        if delta_privacy[i] > 0 and utility_delta > 0:
            pds[i] = 2  # Win-Win
        elif delta_privacy[i] > 0 and utility_delta >= 0:
            pds[i] = 1  # Privacy gain
        elif delta_privacy[i] == 0 and utility_delta == 0:
            pds[i] = 0  # Neutral
        elif delta_privacy[i] < 0 and utility_delta < 0:
            pds[i] = -2  # Lose-Lose
        else:
            pds[i] = -1  # Tradeoff
    return pds

def calculate_pds_gain(privacy_gain, utility_delta):
    """Calculate Pareto Dominance Score based on privacy gain"""
    pds = np.zeros(len(privacy_gain))
    significant_gain = 0.095
    modest_gain = 0.049
    
    for i in range(len(privacy_gain)):
        if privacy_gain[i] > significant_gain and utility_delta > 0:
            pds[i] = 2
        elif privacy_gain[i] > significant_gain and utility_delta == 0:
            pds[i] = 1
        elif privacy_gain[i] > modest_gain and utility_delta > 0:
            pds[i] = 1
        elif (modest_gain >= privacy_gain[i] >= 0 and utility_delta >= 0) or \
             (privacy_gain[i] == 0 and utility_delta == 0):
            pds[i] = 0
        elif privacy_gain[i] < 0 and utility_delta < 0:
            pds[i] = -2
        else:
            pds[i] = -1
    
    return pds

def calculate_wwp(pds):
    """Calculate Win-Win Percentage"""
    win_win_count = np.sum(pds == 2)
    wwp = (win_win_count / len(pds)) * 100
    return wwp

def statistical_significance_test(privacy_gain):
    """Perform t-test for statistical significance"""
    mean_pg = np.mean(privacy_gain)
    se_pg = stats.sem(privacy_gain)
    t_stat = mean_pg / se_pg if se_pg > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(privacy_gain)-1))
    return t_stat, p_value, mean_pg, se_pg

def visualize_privacy_metrics(delta_privacy, privacy_gain, pds_delta, pds_gain, 
                              utility_delta):
    """Visualize privacy metrics"""
    print(f"\n{'='*80}")
    print("GENERATING PRIVACY VISUALIZATIONS")
    print(f"{'='*80}")
    
    seq_length = len(delta_privacy)
    x_pos = np.arange(1, seq_length + 1)
    
    # 1. Privacy Gain
    plt.figure(figsize=(10, 6))
    display_values = (np.exp(privacy_gain) - 1) * 100
    display_values = np.minimum(display_values, 50.0)
    colors = ['green' if x > 0 else 'red' for x in privacy_gain]
    plt.bar(x_pos, display_values, color=colors, edgecolor='black', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
    avg_display = (np.exp(np.mean(privacy_gain)) - 1) * 100
    avg_display = min(avg_display, 50.0)
    plt.axhline(y=avg_display, color='blue', linestyle='--', linewidth=2,
                label=f'Average: {avg_display:.1f}%')
    
    plt.title('Error-Based Privacy Gain by Position', fontsize=14, fontweight='bold')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Privacy Gain (%)', fontsize=12)
    plt.xticks(x_pos)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/privacy_gain.png", dpi=300)
    plt.close()
    
    # 2. PDS Delta
    plt.figure(figsize=(10, 6))
    pds_colors = ['green' if x == 2 else 'lightgreen' if x == 1 else 'gray' if x == 0 
                  else 'lightcoral' if x == -1 else 'red' for x in pds_delta]
    plt.bar(x_pos, pds_delta, color=pds_colors, edgecolor='black', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=np.mean(pds_delta), color='blue', linestyle='--', linewidth=2,
                label=f'Average: {np.mean(pds_delta):.2f}')
    
    plt.title('Pareto Dominance Score (Delta-Privacy)', fontsize=14, fontweight='bold')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('PDS-Delta', fontsize=12)
    plt.xticks(x_pos)
    plt.yticks([-2, -1, 0, 1, 2], ['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy\nGain', 'Win-Win'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_delta.png", dpi=300)
    plt.close()
    
    # 3. PDS Gain
    plt.figure(figsize=(10, 6))
    pds_colors = ['green' if x == 2 else 'lightgreen' if x == 1 else 'gray' if x == 0 
                  else 'lightcoral' if x == -1 else 'red' for x in pds_gain]
    plt.bar(x_pos, pds_gain, color=pds_colors, edgecolor='black', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=np.mean(pds_gain), color='blue', linestyle='--', linewidth=2,
                label=f'Average: {np.mean(pds_gain):.2f}')
    
    plt.title('Pareto Dominance Score (Privacy-Gain)', fontsize=14, fontweight='bold')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('PDS-Gain', fontsize=12)
    plt.xticks(x_pos)
    plt.yticks([-2, -1, 0, 1, 2], ['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy\nGain', 'Win-Win'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_gain.png", dpi=300)
    plt.close()
    
    # 4. PDS Distribution Delta (NEW from Document 3)
    plt.figure(figsize=(10, 6))
    pds_counts_delta = np.array([(pds_delta == -2).sum(), (pds_delta == -1).sum(), 
                                  (pds_delta == 0).sum(), (pds_delta == 1).sum(), (pds_delta == 2).sum()])
    pds_percentage_delta = pds_counts_delta / len(pds_delta) * 100
    pds_categories = ['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy Gain', 'Win-Win']
    colors = ['red', 'lightcoral', 'gray', 'lightgreen', 'green']
    
    bars = plt.bar(pds_categories, pds_percentage_delta, color=colors, edgecolor='black', linewidth=1.5)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pds_percentage_delta[i]:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('PDS Distribution (Delta-Privacy)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage of Positions (%)', fontsize=12)
    plt.ylim(0, max(pds_percentage_delta) * 1.15)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_distribution_delta.png", dpi=300)
    plt.close()
    
    # 5. PDS Distribution Gain (NEW from Document 3)
    plt.figure(figsize=(10, 6))
    pds_counts_gain = np.array([(pds_gain == -2).sum(), (pds_gain == -1).sum(), 
                                 (pds_gain == 0).sum(), (pds_gain == 1).sum(), (pds_gain == 2).sum()])
    pds_percentage_gain = pds_counts_gain / len(pds_gain) * 100
    
    bars = plt.bar(pds_categories, pds_percentage_gain, color=colors, edgecolor='black', linewidth=1.5)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pds_percentage_gain[i]:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('PDS Distribution (Privacy-Gain)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage of Positions (%)', fontsize=12)
    plt.ylim(0, max(pds_percentage_gain) * 1.15)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_distribution_gain.png", dpi=300)
    plt.close()
    
    # Calculate WWP
    wwp_delta = calculate_wwp(pds_delta)
    wwp_gain = calculate_wwp(pds_gain)
    
    print(f"Privacy metrics calculated:")
    print(f"  WWP-Delta: {wwp_delta:.2f}%")
    print(f"  WWP-Gain: {wwp_gain:.2f}%")
    print(f"  Average Privacy Gain: {avg_display:.2f}%")
    print(f"? Privacy visualizations saved")
    print(f"{'='*80}")
    
    return {
        "win_win_percentage_delta": wwp_delta,
        "win_win_percentage_gain": wwp_gain,
        "avg_privacy_gain": avg_display,
        "avg_pds_delta": np.mean(pds_delta),
        "avg_pds_gain": np.mean(pds_gain),
        "utility_gain": utility_delta * 100
    }

# ============================================================================
# ENHANCED VISUALIZATIONS FROM DOCUMENT 3
# ============================================================================

def create_privacy_utility_summary_box(privacy_metrics, t_stat, p_value):
    """Create summary box visualization (NEW from Document 3)"""
    print(f"\nCreating privacy-utility summary box...")
    
    significance = "significant" if p_value < 0.05 else "not significant"
    
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    
    summary_text = [
        "PRIVACY-UTILITY ANALYSIS SUMMARY",
        "="*50,
        "",
        "Delta-Privacy Based Metrics:",
        f"  WWP-Delta: {privacy_metrics['win_win_percentage_delta']:.1f}%",
        f"  Avg PDS-Delta: {privacy_metrics['avg_pds_delta']:.2f}",
        "",
        "Privacy-Gain Based Metrics:",
        f"  WWP-Gain: {privacy_metrics['win_win_percentage_gain']:.1f}%", 
        f"  Avg Privacy Gain: {privacy_metrics['avg_privacy_gain']:.1f}%",
        f"  Avg PDS-Gain: {privacy_metrics['avg_pds_gain']:.2f}",
        "",
        "Utility Metrics:",
        f"  Utility Gain: {privacy_metrics['utility_gain']:.2f}%",
        "",
        "Statistical Testing:",
        "="*50,
        f"  t-statistic: {t_stat:.3f}",
        f"  p-value: {p_value:.6f}",
        f"  Result: Privacy improvement is {significance}",
        "",
        "Interpretation Guide:",
        "="*50,
        "  Win-Win (2):        Both privacy and utility improved",
        "  Privacy Gain (1):   Privacy improved, utility neutral/declined",
        "  Neutral (0):        No significant change",
        "  Tradeoff (-1):      Mixed results",
        "  Lose-Lose (-2):     Both privacy and utility declined"
    ]
    
    plt.text(0.05, 0.95, '\n'.join(summary_text), fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/privacy_utility_summary.png", dpi=300)
    plt.close()
    print(f"? Summary box saved")

def create_position_pareto_frontiers(delta_privacy, privacy_gain, utility_delta):
    """Create Pareto frontier plots (NEW from Document 3)"""
    print(f"\nCreating Pareto frontier visualizations...")
    
    seq_length = len(delta_privacy)
    positions = np.arange(1, seq_length + 1)
    
    # Delta Privacy Frontier
    win_win_delta = np.where((delta_privacy > 0) & (utility_delta > 0))[0]
    privacy_only_delta = np.where((delta_privacy > 0) & (utility_delta <= 0))[0]
    lose_lose_delta = np.where((delta_privacy <= 0) & (utility_delta <= 0))[0]
    utility_only_delta = np.where((delta_privacy <= 0) & (utility_delta > 0))[0]
    
    plt.figure(figsize=(12, 8))
    
    if len(win_win_delta) > 0:
        plt.scatter(np.full(len(win_win_delta), utility_delta*100), 
                   delta_privacy[win_win_delta]*100, s=100, c='green', 
                   alpha=0.7, label='Win-Win', edgecolors='black', linewidth=1)
    if len(privacy_only_delta) > 0:
        plt.scatter(np.full(len(privacy_only_delta), utility_delta*100), 
                   delta_privacy[privacy_only_delta]*100, s=100, c='orange', 
                   alpha=0.7, label='Privacy Only', edgecolors='black', linewidth=1)
    if len(lose_lose_delta) > 0:
        plt.scatter(np.full(len(lose_lose_delta), utility_delta*100), 
                   delta_privacy[lose_lose_delta]*100, s=100, c='red', 
                   alpha=0.7, label='Lose-Lose', edgecolors='black', linewidth=1)
    if len(utility_only_delta) > 0:
        plt.scatter(np.full(len(utility_only_delta), utility_delta*100), 
                   delta_privacy[utility_only_delta]*100, s=100, c='blue', 
                   alpha=0.7, label='Utility Only', edgecolors='black', linewidth=1)
    
    for i, pos in enumerate(positions):
        plt.annotate(str(pos), (utility_delta*100, delta_privacy[i]*100),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    plt.title('Position-Specific Pareto Frontier: Delta Privacy', fontsize=14, fontweight='bold')
    plt.xlabel('Utility Gain (%)', fontsize=12)
    plt.ylabel('Delta Privacy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/position_pareto_frontier_delta.png", dpi=300)
    plt.close()
    
    # Privacy Gain Frontier
    display_values = (np.exp(privacy_gain) - 1) * 100
    display_values = np.minimum(display_values, 50.0)
    
    win_win_gain = np.where((privacy_gain > 0) & (utility_delta > 0))[0]
    privacy_only_gain = np.where((privacy_gain > 0) & (utility_delta <= 0))[0]
    lose_lose_gain = np.where((privacy_gain < 0) & (utility_delta < 0))[0]
    utility_only_gain = np.where((privacy_gain <= 0) & (utility_delta > 0))[0]
    
    plt.figure(figsize=(12, 8))
    
    if len(win_win_gain) > 0:
        plt.scatter(np.full(len(win_win_gain), utility_delta*100), 
                   display_values[win_win_gain], s=100, c='green', 
                   alpha=0.7, label='Win-Win', edgecolors='black', linewidth=1)
    if len(privacy_only_gain) > 0:
        plt.scatter(np.full(len(privacy_only_gain), utility_delta*100), 
                   display_values[privacy_only_gain], s=100, c='orange', 
                   alpha=0.7, label='Privacy Only', edgecolors='black', linewidth=1)
    if len(lose_lose_gain) > 0:
        plt.scatter(np.full(len(lose_lose_gain), utility_delta*100), 
                   display_values[lose_lose_gain], s=100, c='red', 
                   alpha=0.7, label='Lose-Lose', edgecolors='black', linewidth=1)
    if len(utility_only_gain) > 0:
        plt.scatter(np.full(len(utility_only_gain), utility_delta*100), 
                   display_values[utility_only_gain], s=100, c='blue', 
                   alpha=0.7, label='Utility Only', edgecolors='black', linewidth=1)
    
    for i, pos in enumerate(positions):
        plt.annotate(str(pos), (utility_delta*100, display_values[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    plt.title('Position-Specific Pareto Frontier: Privacy Gain', fontsize=14, fontweight='bold')
    plt.xlabel('Utility Gain (%)', fontsize=12)
    plt.ylabel('Privacy Gain (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/position_pareto_frontier_gain.png", dpi=300)
    plt.close()
    
    win_win_delta_positions = [int(positions[i]) for i in win_win_delta]
    win_win_gain_positions = [int(positions[i]) for i in win_win_gain]
    common_win_win = set(win_win_delta_positions).intersection(set(win_win_gain_positions))
    
    print(f"  Win-Win positions (Delta): {win_win_delta_positions}")
    print(f"  Win-Win positions (Gain): {win_win_gain_positions}")
    print(f"  Common Win-Win positions: {list(common_win_win)}")
    print(f"? Pareto frontier plots saved")
    
    return {
        "delta_privacy": {"win_win_positions": win_win_delta_positions},
        "privacy_gain": {"win_win_positions": win_win_gain_positions},
        "common_win_win": list(common_win_win)
    }

def create_pds_detailed_comparison(pds_delta, pds_gain, delta_privacy, privacy_gain):
    """Create detailed PDS comparison (NEW from Document 3)"""
    print(f"\nCreating detailed PDS comparison...")
    
    seq_length = len(pds_delta)
    x_pos = np.arange(1, seq_length + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # PDS-Delta
    colors_delta = ['green' if x == 2 else 'lightgreen' if x == 1 else 'gray' if x == 0 
                    else 'lightcoral' if x == -1 else 'red' for x in pds_delta]
    ax1.bar(x_pos, pds_delta, color=colors_delta, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(y=np.mean(pds_delta), color='blue', linestyle='--', 
                label=f'Avg: {np.mean(pds_delta):.2f}')
    ax1.set_title('PDS-Delta: Raw Accuracy Difference', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PDS-Delta', fontsize=12)
    ax1.set_yticks([-2, -1, 0, 1, 2])
    ax1.set_yticklabels(['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy\nGain', 'Win-Win'])
    ax1.set_xticks(x_pos)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # PDS-Gain
    colors_gain = ['green' if x == 2 else 'lightgreen' if x == 1 else 'gray' if x == 0 
                   else 'lightcoral' if x == -1 else 'red' for x in pds_gain]
    ax2.bar(x_pos, pds_gain, color=colors_gain, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=np.mean(pds_gain), color='blue', linestyle='--', 
                label=f'Avg: {np.mean(pds_gain):.2f}')
    ax2.set_title('PDS-Gain: Logarithmic Error Reduction', fontsize=14, fontweight='bold')
    ax2.set_ylabel('PDS-Gain', fontsize=12)
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax2.set_yticklabels(['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy\nGain', 'Win-Win'])
    ax2.set_xticks(x_pos)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Difference
    difference = pds_delta - pds_gain
    colors_diff = ['darkgreen' if x > 0 else 'darkred' if x < 0 else 'lightgray' for x in difference]
    ax3.bar(x_pos, difference, color=colors_diff, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax3.axhline(y=np.mean(difference), color='purple', linestyle='--',
                label=f'Avg: {np.mean(difference):.2f}')
    ax3.set_title('Difference (PDS-Delta - PDS-Gain)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Position', fontsize=12)
    ax3.set_ylabel('Difference', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_detailed_comparison.png", dpi=300)
    plt.close()
    
    disagreement_positions = x_pos[pds_delta != pds_gain]
    print(f"  Positions where PDS-Delta ? PDS-Gain: {list(disagreement_positions)}")
    print(f"? Detailed PDS comparison saved")
    
    return disagreement_positions

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Create tokenizer
    vocab, k = create_dnagpt_tokenizer(k=6)
    
    # Load data
    train_df, test_df = load_data()
    
    # Create tokenized datasets
    train_dataset, test_dataset = create_datasets(train_df, test_df, vocab, k=k)
    
    # Initialize pretrained (random) model
    pretrained_model = initialize_dnagpt(len(vocab))
    
    # Extract pretrained embeddings
    pretrained_train_emb = extract_embeddings(pretrained_model, 
                                              train_df['sequence'].tolist(), vocab, k)
    pretrained_test_emb = extract_embeddings(pretrained_model, 
                                            test_df['sequence'].tolist(), vocab, k)
    
    # Save pretrained embeddings
    np.save(f"{output_dir}pretrained_train_embeddings.npy", pretrained_train_emb)
    np.save(f"{output_dir}pretrained_test_embeddings.npy", pretrained_test_emb)
    print(f"? Pretrained embeddings saved")
    
    # Train MLP on pretrained embeddings
    _, pretrained_metrics = train_mlp(
        pretrained_train_emb, train_df['label'].values,
        pretrained_test_emb, test_df['label'].values,
        model_type="pretrained"
    )
    
    # Fine-tune DNAGPT
    finetuned_model, _ = finetune_dnagpt(train_dataset, test_dataset, pretrained_model)
    
    # Extract finetuned embeddings
    finetuned_train_emb = extract_embeddings(finetuned_model, 
                                            train_df['sequence'].tolist(), vocab, k)
    finetuned_test_emb = extract_embeddings(finetuned_model, 
                                           test_df['sequence'].tolist(), vocab, k)
    
    # Save finetuned embeddings
    np.save(f"{output_dir}finetuned_train_embeddings.npy", finetuned_train_emb)
    np.save(f"{output_dir}finetuned_test_embeddings.npy", finetuned_test_emb)
    print(f"? Fine-tuned embeddings saved")
    
    # Train MLP on finetuned embeddings
    _, finetuned_metrics = train_mlp(
        finetuned_train_emb, train_df['label'].values,
        finetuned_test_emb, test_df['label'].values,
        model_type="finetuned"
    )
    
    # Create comprehensive visualizations
    pretrained_utility = create_comprehensive_visualizations(
        pretrained_metrics, "Pretrained DNAGPT")
    finetuned_utility = create_comprehensive_visualizations(
        finetuned_metrics, "Fine-tuned DNAGPT")
    
    # Calculate utility delta
    utility_delta = finetuned_utility["accuracy"] - pretrained_utility["accuracy"]
    
    # PCA visualizations
    visualize_embeddings_pca(pretrained_test_emb, test_df['label'].values, 
                            "Pretrained DNAGPT")
    visualize_embeddings_pca(finetuned_test_emb, test_df['label'].values, 
                            "Fine-tuned DNAGPT")
    
    # Reconstruction attacks
    pretrained_attack = run_reconstruction_attack(
        pretrained_test_emb, test_df['sequence'].tolist(), "pretrained")
    finetuned_attack = run_reconstruction_attack(
        finetuned_test_emb, test_df['sequence'].tolist(), "finetuned")
    
    # Compare attacks
    delta_privacy, pretrained_acc, finetuned_acc = compare_reconstruction_attacks(
        pretrained_attack, finetuned_attack)
    
    # Calculate privacy metrics
    privacy_gain = calculate_privacy_gain_log(pretrained_acc, finetuned_acc)
    pds_delta = calculate_pds_delta(delta_privacy, utility_delta)
    pds_gain = calculate_pds_gain(privacy_gain, utility_delta)
    
    # Statistical testing
    t_stat, p_value, _, _ = statistical_significance_test(privacy_gain)
    significance = "significant" if p_value < 0.05 else "not significant"
    
    # Visualize privacy metrics
    privacy_metrics = visualize_privacy_metrics(
        delta_privacy, privacy_gain, pds_delta, pds_gain, utility_delta)
    
    # NEW: Enhanced visualizations from Document 3
    create_privacy_utility_summary_box(privacy_metrics, t_stat, p_value)
    position_analysis = create_position_pareto_frontiers(
        delta_privacy, privacy_gain, utility_delta)
    disagreement_positions = create_pds_detailed_comparison(
        pds_delta, pds_gain, delta_privacy, privacy_gain)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - COMPLETE ANALYSIS")
    print(f"{'='*80}")
    print(f"Model: DNAGPT (Zhang et al. 2023)")
    print(f"Implementation: Custom k-mer tokenization (k=6)")
    print(f"")
    print(f"UTILITY METRICS:")
    print(f"  Pretrained Accuracy:  {pretrained_utility['accuracy']:.4f}")
    print(f"  Fine-tuned Accuracy:  {finetuned_utility['accuracy']:.4f}")
    print(f"  Utility Gain:         {utility_delta:.4f} ({utility_delta*100:.2f}%)")
    print(f"")
    print(f"PRIVACY METRICS:")
    print(f"  Pretrained Attack Acc: {np.mean(pretrained_acc):.4f}")
    print(f"  Fine-tuned Attack Acc: {np.mean(finetuned_acc):.4f}")
    print(f"  Delta Privacy:         {np.mean(delta_privacy):.4f}")
    print(f"  Privacy Gain:          {privacy_metrics['avg_privacy_gain']:.2f}%")
    print(f"")
    print(f"JOINT METRICS:")
    print(f"  WWP-Delta:   {privacy_metrics['win_win_percentage_delta']:.2f}%")
    print(f"  WWP-Gain:    {privacy_metrics['win_win_percentage_gain']:.2f}%")
    print(f"  PDS-Delta:   {privacy_metrics['avg_pds_delta']:.2f}")
    print(f"  PDS-Gain:    {privacy_metrics['avg_pds_gain']:.2f}")
    print(f"")
    print(f"POSITION ANALYSIS:")
    print(f"  Win-Win positions (Delta): {position_analysis['delta_privacy']['win_win_positions']}")
    print(f"  Win-Win positions (Gain):  {position_analysis['privacy_gain']['win_win_positions']}")
    print(f"  Common Win-Win:            {position_analysis['common_win_win']}")
    print(f"  Disagreement positions:    {list(disagreement_positions)}")
    print(f"")
    print(f"STATISTICAL TESTING:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.6f}")
    print(f"  Result:      Privacy improvement is {significance}")
    print(f"{'='*80}")
    print(f"? Complete analysis finished!")
    print(f"? All results saved to: {output_dir}")
    print(f"")
    print(f"TOTAL VISUALIZATIONS GENERATED:")
    print(f"   21 original visualizations")
    print(f"   5 enhanced visualizations from Document 3")
    print(f"   26 TOTAL publication-ready plots")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()