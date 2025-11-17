# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
import matplotlib.ticker as mtick

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Path to your preprocessed HS3D dataset
train_path = 'hs3d_train_pan.csv'
test_path = 'hs3d_test_pan.csv'

# Use XLNet large cased (changed from xlnet-base-cased)
model_name = "xlnet-large-cased"
output_dir = "./xlnet-large-result-double-win/"
os.makedirs(output_dir, exist_ok=True)

# 1. Load and preprocess the HS3D dataset
def load_data():
    print("Loading and preprocessing data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Convert sequences to strings if they're not already
    train_df['sequence'] = train_df['sequence'].astype(str)
    test_df['sequence'] = test_df['sequence'].astype(str)

    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    print(f"Class distribution in training: {train_df['label'].value_counts().to_dict()}")
    print(f"Class distribution in testing: {test_df['label'].value_counts().to_dict()}")

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
    plt.savefig(f"{output_dir}class_distribution.png")
    plt.close()

    return train_df, test_df

# 2. Create HuggingFace datasets
def create_datasets(train_df, test_df):
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = XLNetTokenizer.from_pretrained(model_name)

    # Add nucleotide tokens to tokenizer if they're not already there
    special_tokens = {'additional_special_tokens': ['A', 'C', 'G', 'T']}
    tokenizer.add_special_tokens(special_tokens)

    def tokenize_function(examples):
        # For DNA sequences, tokenize each nucleotide separately
        tokenized = []
        for seq in examples['sequence']:
            # Insert spaces between nucleotides to ensure they're tokenized individually
            spaced_seq = ' '.join(list(seq))
            tokenized.append(spaced_seq)

        return tokenizer(
            tokenized,
            padding='max_length',
            truncation=True,
            max_length=60,  # Longer due to spaces
            return_tensors='pt'
        )

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Visualize sequence length distribution
    seq_lengths = [len(seq) for seq in train_df['sequence']]
    plt.figure(figsize=(10, 6))
    plt.hist(seq_lengths, bins=20, alpha=0.7, color='skyblue')
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}sequence_length_distribution.png")
    plt.close()

    return tokenized_train, tokenized_test, tokenizer

# 3. Fine-tuning XLNet
def finetune_xlnet(tokenized_train, tokenized_test, tokenizer):
    print("Starting fine-tuning of XLNet-Large...")

    # Load pre-trained XLNet for sequence classification
    model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    # Note: XLNet-Large has more parameters, so we may need to adjust batch size
    training_args = TrainingArguments(
        output_dir=f"{output_dir}finetune",
        learning_rate=2e-5,  # Slightly lower learning rate for XLNet-Large
        per_device_train_batch_size=16,  # Reduced batch size due to larger model
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch
        report_to="none",
    )

    # Define compute_metrics function for evaluation
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        auc = None
        try:
            auc = roc_auc_score(labels, pred.predictions[:, 1])
        except:
            pass
        
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc if auc is not None else 0.0
        }

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate
    eval_result = trainer.evaluate()
    print(f"Fine-tuned XLNet-Large evaluation: {eval_result}")

    # Save the model
    model.save_pretrained(f"{output_dir}finetune_xlnet_large")
    tokenizer.save_pretrained(f"{output_dir}finetune_xlnet_large")

    return model, eval_result

# 4. Extract LAST token embeddings from XLNet (key difference from BERT)
def extract_embeddings(model, tokenizer, sequences):
    print(f"Extracting embeddings from XLNet-Large model...")

    # Use the base XLNet model for feature extraction
    feature_extractor = model.transformer
    feature_extractor.eval()

    all_embeddings = []
    batch_size = 16  # Reduced batch size for XLNet-Large

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]

            # Tokenize nucleotides individually with spaces
            tokenized = []
            for seq in batch_sequences:
                spaced_seq = ' '.join(list(seq))
                tokenized.append(spaced_seq)

            inputs = tokenizer(
                tokenized,
                padding='max_length',
                truncation=True,
                max_length=60,
                return_tensors='pt'
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                feature_extractor = feature_extractor.cuda()

            outputs = feature_extractor(**inputs)

            # XLNet: Extract from LAST token position (not CLS like BERT)
            last_hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Get actual sequence lengths
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 because of 0-indexing
            
            # Extract from actual last position for each sequence
            batch_embeddings = []
            for idx, seq_len in enumerate(seq_lengths):
                last_token_embedding = last_hidden_states[idx, seq_len, :].cpu().numpy()
                # For 12-nucleotide sequence ? extracts from position 11 ?
                # For 20-nucleotide sequence ? extracts from position 19 ?
                batch_embeddings.append(last_token_embedding)
            
            all_embeddings.append(np.array(batch_embeddings))

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    return embeddings

# 5. Implement Pan's 3-layer MLP classifier for splice site prediction
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, num_classes=2):
        super(MLPClassifier, self).__init__()
        # Pan et al. used a 3-layer MLP with 200 hidden units and sigmoid activation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization as mentioned in Pan's paper
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.bn1(self.sigmoid(self.fc1(x)))
        x = self.bn2(self.sigmoid(self.fc2(x)))
        x = self.fc3(x)
        return x

# 6. Train and evaluate the MLP classifier
def train_mlp(train_embeddings, train_labels, test_embeddings, test_labels, model_type):
    print(f"Training MLP classifier on {model_type} embeddings...")

    # Create datasets
    train_data = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    test_data = TensorDataset(
        torch.tensor(test_embeddings, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize model
    input_dim = train_embeddings.shape[1]  # Embedding dimension (1024 for XLNet-Large)
    model = MLPClassifier(input_dim=input_dim)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    train_losses = []
    val_accuracies = []
    all_epoch_metrics = []

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

        # Evaluate
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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # Save the model
    torch.save(model.state_dict(), f"{output_dir}mlp_classifier_{model_type}.pt")

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_type} Embeddings')
    plt.tight_layout()
    plt.savefig(f"{output_dir}confusion_matrix_{model_type}.png")
    plt.close()

    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(f"{output_dir}learning_curves_{model_type}.png")
    plt.close()

    # t-SNE visualization
    sample_size = min(1000, len(test_embeddings))
    indices = np.random.choice(len(test_embeddings), sample_size, replace=False)
    sampled_embeddings = test_embeddings[indices]
    sampled_labels = np.array(test_labels)[indices]

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(sampled_embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=sampled_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title(f't-SNE Visualization of {model_type} Embeddings')
    plt.tight_layout()
    plt.savefig(f"{output_dir}tsne_{model_type}.png")
    plt.close()

    return model, all_epoch_metrics

# 7. Create comprehensive visualizations from training metrics
def create_comprehensive_visualizations(all_metrics):
    epochs = [m['epoch'] for m in all_metrics]
    loss = [m['loss'] for m in all_metrics]
    accuracy = [m['accuracy'] for m in all_metrics]
    precision = [m['precision'] for m in all_metrics]
    recall = [m['recall'] for m in all_metrics]
    f1 = [m['f1'] for m in all_metrics]
    auc = [m['auc'] for m in all_metrics]

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

    # Figure 1: Training metrics across epochs with average lines
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracy, 'o-', linewidth=2, label='Accuracy')
    plt.plot(epochs, precision, 'o-', linewidth=2, label='Precision')
    plt.plot(epochs, recall, 'o-', linewidth=2, label='Recall')
    plt.plot(epochs, f1, 'o-', linewidth=2, label='F1 Score')
    plt.plot(epochs, auc, 'o-', linewidth=2, label='AUC')

    plt.axhline(y=avg_accuracy, color='blue', linestyle='--', alpha=0.5, label='Avg Accuracy')
    plt.axhline(y=avg_precision, color='orange', linestyle='--', alpha=0.5, label='Avg Precision')
    plt.axhline(y=avg_recall, color='green', linestyle='--', alpha=0.5, label='Avg Recall')
    plt.axhline(y=avg_f1, color='red', linestyle='--', alpha=0.5, label='Avg F1')
    plt.axhline(y=avg_auc, color='purple', linestyle='--', alpha=0.5, label='Avg AUC')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('XLNet-Large Training Metrics by Epoch', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"{output_dir}training_metrics.png", dpi=300)
    plt.close()

    # Figure 2: Loss vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'o-', color='crimson', linewidth=2)
    plt.axhline(y=avg_loss, color='crimson', linestyle='--', alpha=0.5, label='Avg Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss by Epoch', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}loss_curve.png", dpi=300)
    plt.close()

    # Figure 3: Precision-Recall Trade-off
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, 'o-', color='teal', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Trade-off by Epoch', fontsize=14)

    for i, epoch in enumerate(epochs):
        plt.annotate(f'Epoch {epoch}',
                    (recall[i], precision[i]),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}precision_recall.png", dpi=300)
    plt.close()

    # Figure 4: F1 Score vs Accuracy
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(accuracy, f1, c=epochs, cmap='viridis', s=100, alpha=0.8)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Accuracy by Epoch', fontsize=14)

    for i, epoch in enumerate(epochs):
        plt.annotate(f'Epoch {epoch}',
                    (accuracy[i], f1[i]),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=9)

    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Epoch')
    plt.tight_layout()
    plt.savefig(f"{output_dir}f1_accuracy.png", dpi=300)
    plt.close()

    # Figure 5: Average metrics comparison
    plt.figure(figsize=(10, 6))
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    avg_values = [avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1, avg_auc]
    std_values = [std_loss, std_accuracy, std_precision, std_recall, std_f1, std_auc]

    bars = plt.bar(metrics, avg_values, yerr=std_values, capsize=10, color=sns.color_palette("muted"))

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_values[i] + 0.02,
                f'{avg_values[i]:.4f}', ha='center', va='bottom', rotation=0, fontsize=9)

    plt.ylim(0, 1.1)
    plt.title('Average Performance Metrics Across All Epochs', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}average_metrics.png", dpi=300)
    plt.close()

    # Create a table visualization of all metrics
    plt.figure(figsize=(12, 5))
    plt.axis('off')

    table_data = [
        ['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    ]

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

    plt.title('XLNet-Large Performance Metrics by Epoch', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}metrics_table.png", dpi=300)
    plt.close()

    return {
        "accuracy": avg_accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "auc": avg_auc
    }

# 8. PCA visualization of embeddings
def visualize_embeddings_pca(embeddings, labels, dataset_name):
    pca = PCA(n_components=2)

    sample_size = min(2000, len(embeddings))
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sampled_embeddings = embeddings[indices]
    sampled_labels = labels[indices]

    embeddings_2d = pca.fit_transform(sampled_embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=sampled_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title(f'PCA Visualization of {dataset_name} Embeddings')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}pca_{dataset_name.lower()}_embeddings.png", dpi=300)
    plt.close()

    print(f"PCA on {dataset_name} embeddings:")
    print(f"  PC1 explains {pca.explained_variance_ratio_[0]:.2%} of variance")
    print(f"  PC2 explains {pca.explained_variance_ratio_[1]:.2%} of variance")
    print(f"  Total: {sum(pca.explained_variance_ratio_[:2]):.2%} of variance explained")

# 9. Implementation of Pan's Reconstruction Attack Model
class ReconstructionAttackModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=200):
        super(ReconstructionAttackModel, self).__init__()
        # Pan uses a 3-layer MLP with 200 hidden units and sigmoid activation
        # Input is the concatenation of embedding and positional embedding (2*embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)  # 4 outputs for A, C, G, T
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.bn1(self.sigmoid(self.fc1(x)))
        x = self.bn2(self.sigmoid(self.fc2(x)))
        x = self.fc3(x)
        return x

# 10. Create positional embeddings for the attack model
def create_positional_embedding(position, dim):
    """Create sinusoidal positional embedding as used in Pan's paper"""
    pos_embedding = torch.zeros(dim)
    for i in range(0, dim, 2):
        pos_embedding[i] = np.sin(position / (10000 ** (i / dim)))
        if i + 1 < dim:
            pos_embedding[i + 1] = np.cos(position / (10000 ** ((i + 1) / dim)))
    return pos_embedding

# 11. Implementation of Pan's reconstruction attack
def run_reconstruction_attack(embeddings, sequences, model_name, output_dir="./xlnet-large-result-double-win/"):
    """Implement Pan's reconstruction attack on embeddings"""
    print(f"Running reconstruction attack on {model_name} embeddings...")
    os.makedirs(f"{output_dir}attack_results/", exist_ok=True)

    embedding_dim = embeddings.shape[1]
    seq_length = 20  # As shown in Pan's paper Fig. 3

    valid_indices = [i for i, seq in enumerate(sequences) if len(seq) >= seq_length]
    if len(valid_indices) < len(sequences):
        print(f"Warning: {len(sequences) - len(valid_indices)} sequences are shorter than {seq_length}")
        embeddings = embeddings[valid_indices]
        sequences = [sequences[i] for i in valid_indices]

    nuc_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    int_to_nuc = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    position_accuracy = []
    confusion_matrices = []

    for pos in range(seq_length):
        print(f"Training attack model for position {pos+1}/{seq_length}")

        pos_embeddings = torch.stack([create_positional_embedding(pos, embedding_dim)
                                    for _ in range(len(embeddings))])

        y_true = torch.tensor([nuc_to_int.get(seq[pos].upper(), 0) for seq in sequences])

        X = torch.cat([torch.tensor(embeddings, dtype=torch.float32),
                      pos_embeddings], dim=1)

        dataset = TensorDataset(X, y_true)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model = ReconstructionAttackModel(embedding_dim)
        if torch.cuda.is_available():
            model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 5
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
            if (epoch + 1) % 1 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

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

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['A', 'C', 'G', 'T'],
                    yticklabels=['A', 'C', 'G', 'T'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Position {pos+1}')
        plt.tight_layout()
        plt.savefig(f"{output_dir}attack_results/confusion_matrix_{model_name}_pos_{pos+1}.png")
        plt.close()

        print(f"Position {pos+1} accuracy: {acc:.4f}")

    plt.figure(figsize=(10, 6))
    x_pos = np.arange(1, seq_length + 1)
    plt.plot(x_pos, position_accuracy, 'o-', linewidth=2, color='blue', label=model_name)
    plt.axhline(y=np.mean(position_accuracy), color='red', linestyle='--',
                label=f'Avg: {np.mean(position_accuracy):.3f}')
    plt.axhline(y=0.25, color='gray', linestyle=':', label='Random (0.25)')
    plt.title(f'Accuracy of Reconstruction Attack per Nucleotide Position ({model_name})', fontsize=14)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(x_pos)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/reconstruction_accuracy_{model_name}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(15, 10))
    nrows, ncols = 5, 4
    for i, cm in enumerate(confusion_matrices):
        if i < seq_length:
            plt.subplot(nrows, ncols, i + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['A', 'C', 'G', 'T'],
                        yticklabels=['A', 'C', 'G', 'T'])
            plt.title(f'Position {i+1}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/all_confusion_matrices_{model_name}.png", dpi=300)
    plt.close()

    results_df = pd.DataFrame({
        'Position': list(range(1, seq_length + 1)),
        'Accuracy': position_accuracy
    })
    results_df.to_csv(f"{output_dir}attack_results/reconstruction_results_{model_name}.csv", index=False)

    class_accuracy = {}
    for nuc in ['A', 'C', 'G', 'T']:
        nuc_idx = nuc_to_int[nuc]
        accuracies = []
        for pos in range(seq_length):
            cm = confusion_matrices[pos]
            if cm[nuc_idx].sum() > 0:
                acc = cm[nuc_idx, nuc_idx] / cm[nuc_idx].sum()
                accuracies.append(acc)
        if accuracies:
            class_accuracy[nuc] = np.mean(accuracies)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_accuracy.keys(), class_accuracy.values(), color='teal')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    plt.axhline(y=0.25, color='red', linestyle='--', label='Random (0.25)')
    plt.ylim(0, 1.0)
    plt.title(f'Nucleotide-wise Reconstruction Accuracy ({model_name})', fontsize=14)
    plt.ylabel('Average Accuracy', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/nucleotide_accuracy_{model_name}.png", dpi=300)
    plt.close()

    print(f"Average reconstruction accuracy: {np.mean(position_accuracy):.4f}")
    print(f"Nucleotide-wise accuracy: {class_accuracy}")
    return position_accuracy, confusion_matrices, class_accuracy

# 12. Compare reconstruction attacks
def compare_reconstruction_attacks(pretrained_results, finetuned_results, output_dir="./xlnet-large-result-double-win/"):
    """Compare reconstruction attack results between pretrained and fine-tuned models"""
    pretrained_acc, finetuned_acc = pretrained_results[0], finetuned_results[0]

    plt.figure(figsize=(12, 7))
    x_pos = np.arange(1, len(pretrained_acc) + 1)

    plt.plot(x_pos, pretrained_acc, 'o-', linewidth=2, color='blue', label=f'Pretrained (Avg: {np.mean(pretrained_acc):.3f})')
    plt.plot(x_pos, finetuned_acc, 'o-', linewidth=2, color='red', label=f'Fine-tuned (Avg: {np.mean(finetuned_acc):.3f})')

    plt.axhline(y=0.25, color='gray', linestyle=':', label='Random (0.25)')

    for i in range(len(pretrained_acc)):
        if finetuned_acc[i] < pretrained_acc[i]:
            plt.axvspan(i+0.5, i+1.5, alpha=0.2, color='green')

    plt.title('Comparison of Reconstruction Attack Accuracy: Pretrained vs. Fine-tuned', fontsize=14)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(x_pos)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/reconstruction_comparison.png", dpi=300)
    plt.close()

    pretrained_nuc_acc, finetuned_nuc_acc = pretrained_results[2], finetuned_results[2]

    plt.figure(figsize=(10, 6))
    nucleotides = ['A', 'C', 'G', 'T']
    pretrained_values = [pretrained_nuc_acc.get(nuc, 0) for nuc in nucleotides]
    finetuned_values = [finetuned_nuc_acc.get(nuc, 0) for nuc in nucleotides]

    x = np.arange(len(nucleotides))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pretrained_values, width, label='Pretrained')
    rects2 = ax.bar(x + width/2, finetuned_values, width, label='Fine-tuned')

    ax.set_ylabel('Average Accuracy')
    ax.set_title('Nucleotide-wise Reconstruction Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(nucleotides)
    ax.axhline(y=0.25, color='red', linestyle='--', label='Random (0.25)')
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(f"{output_dir}attack_results/nucleotide_comparison.png", dpi=300)
    plt.close()

    delta_privacy = np.array(pretrained_acc) - np.array(finetuned_acc)
    avg_privacy_change = np.mean(delta_privacy)

    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, delta_privacy, color=['green' if x > 0 else 'red' for x in delta_privacy])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=avg_privacy_change, color='blue', linestyle='--',
                label=f'Avg change: {avg_privacy_change:.3f}')

    plt.title('Privacy Change After Fine-tuning (Positive = More Private)', fontsize=14)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Accuracy Difference (Pretrained - Fine-tuned)', fontsize=12)
    plt.xticks(x_pos)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/privacy_change.png", dpi=300)
    plt.close()

    summary_data = {
        'Model': ['Pretrained', 'Fine-tuned'],
        'Avg Accuracy': [np.mean(pretrained_acc), np.mean(finetuned_acc)],
        'Min Accuracy': [np.min(pretrained_acc), np.min(finetuned_acc)],
        'Max Accuracy': [np.max(pretrained_acc), np.max(finetuned_acc)],
        'A Accuracy': [pretrained_nuc_acc.get('A', 0), finetuned_nuc_acc.get('A', 0)],
        'C Accuracy': [pretrained_nuc_acc.get('C', 0), finetuned_nuc_acc.get('C', 0)],
        'G Accuracy': [pretrained_nuc_acc.get('G', 0), finetuned_nuc_acc.get('G', 0)],
        'T Accuracy': [pretrained_nuc_acc.get('T', 0), finetuned_nuc_acc.get('T', 0)],
        'Privacy Change': ['N/A', f"{avg_privacy_change:.4f}"]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}attack_results/comparison_summary.csv", index=False)

    plt.figure(figsize=(12, 4))
    plt.axis('off')

    table = plt.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for i in range(len(summary_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')

    plt.title('Reconstruction Attack Comparison Summary', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/comparison_summary.png", dpi=300)
    plt.close()

    print("\n=== Reconstruction Attack Comparison ===")
    print(f"Pretrained Average Accuracy: {np.mean(pretrained_acc):.4f}")
    print(f"Fine-tuned Average Accuracy: {np.mean(finetuned_acc):.4f}")
    print(f"Privacy Change: {avg_privacy_change:.4f}")
    if avg_privacy_change > 0:
        print("Fine-tuning IMPROVED privacy (lower reconstruction accuracy)")
    else:
        print("Fine-tuning REDUCED privacy (higher reconstruction accuracy)")
        
    return delta_privacy, pretrained_acc, finetuned_acc

# 13. Calculate Error-Based Privacy Gain with true logarithmic transformation
def calculate_privacy_gain_log(pretrained_acc, finetuned_acc):
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

# 14. Calculate Pareto Dominance Score based on delta privacy (PDS-Delta)
def calculate_pds_delta(delta_privacy, utility_delta):
    """Calculate Pareto Dominance Score for each position based on delta_privacy"""
    pds = np.zeros(len(delta_privacy))
    
    for i in range(len(delta_privacy)):
        if delta_privacy[i] > 0 and utility_delta > 0:
            pds[i] = 2
        elif delta_privacy[i] > 0 and utility_delta >= 0:
            pds[i] = 1
        elif delta_privacy[i] == 0 and utility_delta == 0:
            pds[i] = 0
        elif delta_privacy[i] < 0 and utility_delta < 0:
            pds[i] = -2
        else:
            pds[i] = -1
            
    return pds

# 15. Calculate Pareto Dominance Score based on privacy gain (PDS-Gain)
def calculate_pds_gain(privacy_gain, utility_delta):
    """Calculate Pareto Dominance Score for each position based on privacy_gain"""
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
        elif (modest_gain >= privacy_gain[i] >= 0 and utility_delta >= 0) or (privacy_gain[i] == 0 and utility_delta == 0):
            pds[i] = 0
        elif privacy_gain[i] < 0 and utility_delta < 0:
            pds[i] = -2
        else:
            pds[i] = -1
    
    return pds

# 16. Calculate Win-Win Percentage (WWP)
def calculate_wwp(pds):
    win_win_count = np.sum(pds == 2)
    total_positions = len(pds)
    wwp = (win_win_count / total_positions) * 100
    return wwp

# 17. Statistical Significance Testing
def statistical_significance_test(privacy_gain):
    """Perform t-test to determine if privacy gain is significant"""
    mean_pg = np.mean(privacy_gain)
    se_pg = stats.sem(privacy_gain)
    t_stat = mean_pg / se_pg
    
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(privacy_gain)-1))
    
    return t_stat, p_value, mean_pg, se_pg

# 18. Visualize Privacy Gain and PDS
def visualize_privacy_metrics(delta_privacy, privacy_gain, pds_delta, pds_gain, utility_delta, output_dir):
    """Visualize the novel privacy metrics introduced in the paper"""
    seq_length = len(delta_privacy)
    x_pos = np.arange(1, seq_length + 1)
    
    # 1. Plot Privacy Gain
    plt.figure(figsize=(10, 6))
    display_values = (np.exp(privacy_gain) - 1) * 100
    display_values = np.minimum(display_values, 50.0)
    plt.bar(x_pos, display_values, color=['green' if x > 0 else 'red' for x in privacy_gain])
    plt.axhline(y=0, color='black', linestyle='-')
    avg_display = (np.exp(np.mean(privacy_gain)) - 1) * 100
    avg_display = min(avg_display, 50.0) 
    plt.axhline(y=avg_display, color='blue', linestyle='--',
                label=f'Avg gain: {avg_display:.1f}%')
    
    plt.title('Error-Based Privacy Gain by Position (Positive = More Private)', fontsize=14)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Privacy Gain (%)', fontsize=12)
    plt.xticks(x_pos)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/privacy_gain.png", dpi=300)
    plt.close()
    
    # 2. Plot Pareto Dominance Score - Delta Privacy
    plt.figure(figsize=(10, 6))
    colors = ['green' if x == 2 else 
              'lightgreen' if x == 1 else 
              'gray' if x == 0 else 
              'lightcoral' if x == -1 else 
              'red' for x in pds_delta]
    
    plt.bar(x_pos, pds_delta, color=colors)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=np.mean(pds_delta), color='blue', linestyle='--',
                label=f'Avg PDS-Delta: {np.mean(pds_delta):.2f}')
    
    plt.title('Pareto Dominance Score by Position (Delta-Privacy)', fontsize=14)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('PDS-Delta', fontsize=12)
    plt.xticks(x_pos)
    plt.yticks([-2, -1, 0, 1, 2], ['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy Gain', 'Win-Win'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pareto_dominance_score_delta.png", dpi=300)
    plt.close()
    
    # 3. Plot Pareto Dominance Score - Privacy Gain
    plt.figure(figsize=(10, 6))
    colors = ['green' if x == 2 else 
              'lightgreen' if x == 1 else 
              'gray' if x == 0 else 
              'lightcoral' if x == -1 else 
              'red' for x in pds_gain]
    
    plt.bar(x_pos, pds_gain, color=colors)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=np.mean(pds_gain), color='blue', linestyle='--',
                label=f'Avg PDS-Gain: {np.mean(pds_gain):.2f}')
    
    plt.title('Pareto Dominance Score by Position (Privacy-Gain)', fontsize=14)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('PDS-Gain', fontsize=12)
    plt.xticks(x_pos)
    plt.yticks([-2, -1, 0, 1, 2], ['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy Gain', 'Win-Win'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pareto_dominance_score_gain.png", dpi=300)
    plt.close()
    
    # 4. Create a distribution of PDS values - Delta Privacy
    plt.figure(figsize=(10, 6))
    pds_counts_delta = np.array([(pds_delta == -2).sum(), (pds_delta == -1).sum(), 
                          (pds_delta == 0).sum(), (pds_delta == 1).sum(), 
                          (pds_delta == 2).sum()])
    pds_percentage_delta = pds_counts_delta / len(pds_delta) * 100
    pds_categories = ['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy Gain', 'Win-Win']
    colors = ['red', 'lightcoral', 'gray', 'lightgreen', 'green']
    
    bars = plt.bar(pds_categories, pds_percentage_delta, color=colors)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pds_percentage_delta[i]:.1f}%', ha='center', va='bottom')
    
    plt.title('Distribution of Pareto Dominance Scores (Delta-Privacy)', fontsize=14)
    plt.ylabel('Percentage of Positions (%)', fontsize=12)
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_distribution_delta.png", dpi=300)
    plt.close()
    
    # 5. Create a distribution of PDS values - Privacy Gain
    plt.figure(figsize=(10, 6))
    pds_counts_gain = np.array([(pds_gain == -2).sum(), (pds_gain == -1).sum(), 
                          (pds_gain == 0).sum(), (pds_gain == 1).sum(), 
                          (pds_gain == 2).sum()])
    pds_percentage_gain = pds_counts_gain / len(pds_gain) * 100
    
    bars = plt.bar(pds_categories, pds_percentage_gain, color=colors)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pds_percentage_gain[i]:.1f}%', ha='center', va='bottom')
    
    plt.title('Distribution of Pareto Dominance Scores (Privacy-Gain)', fontsize=14)
    plt.ylabel('Percentage of Positions (%)', fontsize=12)
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_distribution_gain.png", dpi=300)
    plt.close()
    
    # 6. Create summary box
    wwp_delta = calculate_wwp(pds_delta)
    wwp_gain = calculate_wwp(pds_gain)
    
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    
    summary_text = [
        "Privacy-Utility Analysis Summary",
        "================================",
        "Delta-Privacy Based Metrics:",
        f"Win-Win Percentage (WWP-Delta): {wwp_delta:.1f}%",
        f"Average Pareto Dominance Score (PDS-Delta): {np.mean(pds_delta):.2f}",
        "",
        "Privacy-Gain Based Metrics:",
        f"Win-Win Percentage (WWP-Gain): {wwp_gain:.1f}%", 
        f"Average Privacy Gain: {avg_display:.1f}%",
        f"Average Pareto Dominance Score (PDS-Gain): {np.mean(pds_gain):.2f}",
        f"Utility Gain: {utility_delta*100:.1f}%"
    ]
    
    t_stat, p_value, mean_pg, se_pg = statistical_significance_test(privacy_gain)
    significance = "significant" if p_value < 0.05 else "not significant"
    
    summary_text.extend([
        "",
        "Statistical Testing",
        "===================",
        f"Privacy Gain t-statistic: {t_stat:.2f}",
        f"p-value: {p_value:.4f}",
        f"Privacy improvement is statistically {significance}",
        "",
        "Interpretation",
        "==============",
        "Win-Win: Both privacy and utility improved",
        "Privacy Gain: Privacy improved with neutral utility",
        "Tradeoff: Either privacy or utility decreased",
        "Lose-Lose: Both privacy and utility decreased"
    ])
    
    plt.text(0.1, 0.95, '\n'.join(summary_text), fontsize=12, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/privacy_utility_summary.png", dpi=300)
    plt.close()
    
    return {
        "win_win_percentage_delta": wwp_delta,
        "win_win_percentage_gain": wwp_gain,
        "avg_privacy_gain": avg_display,
        "avg_pds_delta": np.mean(pds_delta),
        "avg_pds_gain": np.mean(pds_gain),
        "utility_gain": utility_delta * 100,
        "t_statistic": t_stat,
        "p_value": p_value
    }

# 19. Create Pareto Frontier Plots
def create_position_pareto_frontiers(delta_privacy, privacy_gain, utility_delta, output_dir):
    """Create Pareto frontier plots for positions using both metrics"""
    seq_length = len(delta_privacy)
    positions = np.arange(1, seq_length + 1)
    
    # 1. Delta Privacy Pareto Frontier
    win_win_delta_indices = np.where((delta_privacy > 0) & (utility_delta > 0))[0]
    privacy_only_delta_indices = np.where((delta_privacy > 0) & (utility_delta <= 0))[0]
    lose_lose_delta_indices = np.where((delta_privacy <= 0) & (utility_delta <= 0))[0]
    utility_only_delta_indices = np.where((delta_privacy <= 0) & (utility_delta > 0))[0]
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(np.full(len(win_win_delta_indices), utility_delta*100), 
               delta_privacy[win_win_delta_indices]*100, s=100, c='green', alpha=0.7, label='Win-Win')
    plt.scatter(np.full(len(privacy_only_delta_indices), utility_delta*100), 
               delta_privacy[privacy_only_delta_indices]*100, s=100, c='orange', alpha=0.7, label='Privacy Only')
    plt.scatter(np.full(len(lose_lose_delta_indices), utility_delta*100), 
               delta_privacy[lose_lose_delta_indices]*100, s=100, c='red', alpha=0.7, label='Lose-Lose')
    plt.scatter(np.full(len(utility_only_delta_indices), utility_delta*100), 
               delta_privacy[utility_only_delta_indices]*100, s=100, c='blue', alpha=0.7, label='Utility Only')
    
    for i, pos in enumerate(positions):
        plt.annotate(str(pos), (utility_delta*100, delta_privacy[i]*100),
                    xytext=(5, 5), textcoords='offset points')
    
    top_delta_positions_indices = np.argsort(delta_privacy)[-5:]
    
    for i in top_delta_positions_indices:
        plt.scatter(utility_delta*100, delta_privacy[i]*100, s=150, 
                   edgecolor='black', linewidth=2, facecolor='none')
        plt.annotate(f"Pos {positions[i]}", 
                    (utility_delta*100, delta_privacy[i]*100),
                    xytext=(10, 10), textcoords='offset points',
                    fontweight='bold')
    
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    plt.title('Position-Specific Analysis: Delta Privacy', fontsize=14)
    plt.xlabel('Utility Gain (%)', fontsize=12)
    plt.ylabel('Delta Privacy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/position_pareto_frontier_delta.png", dpi=300)
    plt.close()
    
    # 2. Privacy Gain Pareto Frontier
    display_values = (np.exp(privacy_gain) - 1) * 100
    display_values = np.minimum(display_values, 50.0)
    
    win_win_gain_indices = np.where((privacy_gain > 0) & (utility_delta > 0))[0]
    privacy_only_gain_indices = np.where((privacy_gain > 0) & (utility_delta <= 0))[0]
    lose_lose_gain_indices = np.where((privacy_gain < 0) & (utility_delta < 0))[0]
    utility_only_gain_indices = np.where((privacy_gain <= 0) & (utility_delta > 0))[0]
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(np.full(len(win_win_gain_indices), utility_delta*100), 
               display_values[win_win_gain_indices], s=100, c='green', alpha=0.7, label='Win-Win')
    plt.scatter(np.full(len(privacy_only_gain_indices), utility_delta*100), 
               display_values[privacy_only_gain_indices], s=100, c='orange', alpha=0.7, label='Privacy Only')
    plt.scatter(np.full(len(lose_lose_gain_indices), utility_delta*100), 
               display_values[lose_lose_gain_indices], s=100, c='red', alpha=0.7, label='Lose-Lose')
    plt.scatter(np.full(len(utility_only_gain_indices), utility_delta*100), 
               display_values[utility_only_gain_indices], s=100, c='blue', alpha=0.7, label='Utility Only')
    
    for i, pos in enumerate(positions):
        plt.annotate(str(pos), (utility_delta*100, display_values[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    top_gain_positions_indices = np.argsort(privacy_gain)[-5:]
    
    for i in top_gain_positions_indices:
        plt.scatter(utility_delta*100, display_values[i], s=150, 
                   edgecolor='black', linewidth=2, facecolor='none')
        plt.annotate(f"Pos {positions[i]}", 
                    (utility_delta*100, display_values[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontweight='bold')
    
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    plt.title('Position-Specific Analysis: Privacy Gain', fontsize=14)
    plt.xlabel('Utility Gain (%)', fontsize=12)
    plt.ylabel('Privacy Gain (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/position_pareto_frontier_gain.png", dpi=300)
    plt.close()
    
    # Position ranking tables
    sorted_delta_indices = np.argsort(delta_privacy)[::-1]
    sorted_delta_positions = positions[sorted_delta_indices]
    sorted_delta_values = delta_privacy[sorted_delta_indices] * 100
    
    sorted_gain_indices = np.argsort(privacy_gain)[::-1]
    sorted_gain_positions = positions[sorted_gain_indices]
    sorted_gain_values = (np.exp(privacy_gain[sorted_gain_indices]) - 1) * 100
    sorted_gain_values = np.minimum(sorted_gain_values, 50.0)
    
    plt.figure(figsize=(15, 8))
    plt.axis('off')
    
    table_data = [
        ['Rank', 'Delta Privacy', '', 'Privacy Gain', ''],
        ['', 'Position', 'Delta (%)', 'Position', 'Gain (%)'],
    ]
    
    for i in range(min(20, len(sorted_delta_positions))):
        delta_pos = int(sorted_delta_positions[i])
        delta_value = sorted_delta_values[i]
        
        gain_pos = int(sorted_gain_positions[i])
        gain_value = sorted_gain_values[i]
        
        table_data.append([
            f'{i+1}', 
            f'{delta_pos}', f'{delta_value:.2f}%', 
            f'{gain_pos}', f'{gain_value:.2f}%'
        ])
    
    table = plt.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i <= 1:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
    
    plt.title('Position Rankings: Delta Privacy vs Privacy Gain', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/position_ranking_comparison.png", dpi=300)
    plt.close()
    
    win_win_delta_positions = [int(positions[i]) for i in win_win_delta_indices]
    win_win_gain_positions = [int(positions[i]) for i in win_win_gain_indices]
    
    common_win_win = set(win_win_delta_positions).intersection(set(win_win_gain_positions))
    
    return {
        "delta_privacy": {
            "optimal_positions": [int(sorted_delta_positions[i]) for i in range(min(5, len(sorted_delta_positions)))],
            "win_win_positions": win_win_delta_positions,
            "privacy_only_positions": [int(positions[i]) for i in privacy_only_delta_indices],
            "lose_lose_positions": [int(positions[i]) for i in lose_lose_delta_indices],
            "utility_only_positions": [int(positions[i]) for i in utility_only_delta_indices],
        },
        "privacy_gain": {
            "optimal_positions": [int(sorted_gain_positions[i]) for i in range(min(5, len(sorted_gain_positions)))],
            "win_win_positions": win_win_gain_positions,
            "privacy_only_positions": [int(positions[i]) for i in privacy_only_gain_indices],
            "lose_lose_positions": [int(positions[i]) for i in lose_lose_gain_indices],
            "utility_only_positions": [int(positions[i]) for i in utility_only_gain_indices],
        },
        "common_win_win": list(common_win_win)
    }

# 20. Create PDS comparison
def create_pds_comparison(pds_delta, pds_gain, delta_privacy, privacy_gain, output_dir):
    """Create enhanced visualizations that highlight differences between PDS metrics"""
    seq_length = len(pds_delta)
    x_pos = np.arange(1, seq_length + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot 1: PDS-Delta
    colors_delta = ['green' if x == 2 else 
                    'lightgreen' if x == 1 else 
                    'gray' if x == 0 else 
                    'lightcoral' if x == -1 else 
                    'red' for x in pds_delta]
    
    ax1.bar(x_pos, pds_delta, color=colors_delta)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(y=np.mean(pds_delta), color='blue', linestyle='--',
                label=f'Avg: {np.mean(pds_delta):.2f}')
    ax1.set_title('PDS-Delta: Based on Raw Accuracy Difference', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PDS-Delta', fontsize=12)
    ax1.set_yticks([-2, -1, 0, 1, 2])
    ax1.set_yticklabels(['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy\nGain', 'Win-Win'])
    ax1.set_xticks(x_pos)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Plot 2: PDS-Gain
    colors_gain = ['green' if x == 2 else 
                   'lightgreen' if x == 1 else 
                   'gray' if x == 0 else 
                   'lightcoral' if x == -1 else 
                   'red' for x in pds_gain]
    
    ax2.bar(x_pos, pds_gain, color=colors_gain)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=np.mean(pds_gain), color='blue', linestyle='--',
                label=f'Avg: {np.mean(pds_gain):.2f}')
    ax2.set_title('PDS-Gain: Based on Logarithmic Error Reduction', fontsize=14, fontweight='bold')
    ax2.set_ylabel('PDS-Gain', fontsize=12)
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax2.set_yticklabels(['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy\nGain', 'Win-Win'])
    ax2.set_xticks(x_pos)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Plot 3: Difference map
    difference = pds_delta - pds_gain
    colors_diff = ['darkgreen' if x > 0 else 
                   'darkred' if x < 0 else 
                   'lightgray' for x in difference]
    
    ax3.bar(x_pos, difference, color=colors_diff, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax3.set_title('Difference (PDS-Delta - PDS-Gain): Where Metrics Disagree', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Position', fontsize=12)
    ax3.set_ylabel('Difference', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, diff in enumerate(difference):
        if diff != 0:
            ax3.text(x_pos[i], diff, f'Pos {x_pos[i]}', 
                    ha='center', va='bottom' if diff > 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_detailed_comparison.png", dpi=300)
    plt.close()
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    pds_matrix = np.vstack([pds_delta, pds_gain])
    cmap = plt.cm.RdYlGn
    
    im = ax.imshow(pds_matrix, cmap=cmap, aspect='auto', vmin=-2, vmax=2)
    
    ax.set_xticks(np.arange(seq_length))
    ax.set_xticklabels(x_pos)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['PDS-Delta', 'PDS-Gain'])
    
    for i in range(2):
        for j in range(seq_length):
            value = pds_matrix[i, j]
            labels = {-2: 'LL', -1: 'T', 0: 'N', 1: 'PG', 2: 'WW'}
            text = ax.text(j, i, labels[int(value)],
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('PDS Metrics Comparison Heatmap\nWW=Win-Win, PG=Privacy Gain, N=Neutral, T=Tradeoff, LL=Lose-Lose',
                fontsize=14)
    ax.set_xlabel('Position', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_ticklabels(['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy Gain', 'Win-Win'])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_heatmap_comparison.png", dpi=300)
    plt.close()
    
    # Summary table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    win_win_delta = (pds_delta == 2).sum()
    win_win_gain = (pds_gain == 2).sum()
    privacy_only_delta = (pds_delta == 1).sum()
    privacy_only_gain = (pds_gain == 1).sum()
    tradeoff_delta = (pds_delta == -1).sum()
    tradeoff_gain = (pds_gain == -1).sum()
    
    disagreement_positions = x_pos[pds_delta != pds_gain]
    
    table_data = [
        ['Metric', 'Win-Win', 'Privacy Gain', 'Neutral', 'Tradeoff', 'Lose-Lose', 'Avg PDS'],
        ['PDS-Delta', f'{win_win_delta}', f'{privacy_only_delta}', 
         f'{(pds_delta == 0).sum()}', f'{tradeoff_delta}', f'{(pds_delta == -2).sum()}',
         f'{np.mean(pds_delta):.2f}'],
        ['PDS-Gain', f'{win_win_gain}', f'{privacy_only_gain}',
         f'{(pds_gain == 0).sum()}', f'{tradeoff_gain}', f'{(pds_gain == -2).sum()}',
         f'{np.mean(pds_gain):.2f}'],
        ['Difference', f'{win_win_delta - win_win_gain:+d}', 
         f'{privacy_only_delta - privacy_only_gain:+d}',
         f'{(pds_delta == 0).sum() - (pds_gain == 0).sum():+d}',
         f'{tradeoff_delta - tradeoff_gain:+d}',
         f'{(pds_delta == -2).sum() - (pds_gain == -2).sum():+d}',
         f'{np.mean(pds_delta) - np.mean(pds_gain):+.2f}']
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(len(table_data[0])):
        table[(3, i)].set_facecolor('#FFF2CC')
        table[(3, i)].set_text_props(fontweight='bold')
    
    ax.text(0.5, 0.95, f'Positions where metrics disagree: {list(disagreement_positions)}',
            ha='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    plt.title('PDS Metrics Comparison Summary', fontsize=14, pad=40)
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_comparison_summary.png", dpi=300)
    plt.close()
    
    print(f"Created enhanced comparison visualizations in {output_dir}attack_results/")
    print(f"Positions where PDS-Delta and PDS-Gain disagree: {list(disagreement_positions)}")
    
    return disagreement_positions

# 21. Main execution function
def main():
    print("Starting XLNet-Large fine-tuning, embedding extraction, and privacy analysis pipeline")
    print(f"Using XLNet model: {model_name}")
    print(f"Output directory: {output_dir}")

    # Load and preprocess data
    train_df, test_df = load_data()
    tokenized_train, tokenized_test, tokenizer = create_datasets(train_df, test_df)

    # Load pretrained XLNet model
    print("Loading pretrained XLNet-Large model...")
    pretrained_model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2)
    pretrained_model.resize_token_embeddings(len(tokenizer))

    # Extract pretrained embeddings
    print("Extracting embeddings from pretrained model...")
    pretrained_train_embeddings = extract_embeddings(
        pretrained_model, tokenizer, train_df['sequence'].tolist()
    )
    pretrained_test_embeddings = extract_embeddings(
        pretrained_model, tokenizer, test_df['sequence'].tolist()
    )

    # Train MLP on pretrained embeddings
    print("\n=== Training MLP on Pretrained Embeddings ===")
    _, pretrained_metrics = train_mlp(
        pretrained_train_embeddings, train_df['label'].values,
        pretrained_test_embeddings, test_df['label'].values,
        model_type="pretrained_xlnet_large"
    )

    # Save pretrained embeddings
    np.save(f"{output_dir}pretrained_train_embeddings.npy", pretrained_train_embeddings)
    np.save(f"{output_dir}pretrained_test_embeddings.npy", pretrained_test_embeddings)
    print(f"Saved pretrained embeddings to {output_dir}")

    # Fine-tune the model
    finetuned_model, xlnet_eval_results = finetune_xlnet(tokenized_train, tokenized_test, tokenizer)

    # Extract fine-tuned embeddings
    print("Extracting embeddings from fine-tuned model...")
    finetuned_train_embeddings = extract_embeddings(
        finetuned_model, tokenizer, train_df['sequence'].tolist()
    )
    finetuned_test_embeddings = extract_embeddings(
        finetuned_model, tokenizer, test_df['sequence'].tolist()
    )

    # Save fine-tuned embeddings
    np.save(f"{output_dir}finetuned_train_embeddings.npy", finetuned_train_embeddings)
    np.save(f"{output_dir}finetuned_test_embeddings.npy", finetuned_test_embeddings)
    print(f"Saved fine-tuned embeddings to {output_dir}")

    # Train MLP on fine-tuned embeddings
    print("\n=== Training MLP on Fine-tuned Embeddings ===")
    _, finetuned_metrics = train_mlp(
        finetuned_train_embeddings, train_df['label'].values,
        finetuned_test_embeddings, test_df['label'].values,
        model_type="finetuned_xlnet_large"
    )

    # Generate comprehensive visualizations
    pretrained_utility_metrics = create_comprehensive_visualizations(pretrained_metrics)
    finetuned_utility_metrics = create_comprehensive_visualizations(finetuned_metrics)

    # Calculate utility delta
    utility_delta = finetuned_utility_metrics["accuracy"] - pretrained_utility_metrics["accuracy"]
    print(f"Utility delta (fine-tuned - pretrained): {utility_delta:.4f}")

    # Run reconstruction attacks
    print("\n=== Running Reconstruction Attack on Pretrained Embeddings ===")
    pretrained_results = run_reconstruction_attack(
        pretrained_test_embeddings, test_df['sequence'].tolist(),
        "pretrained", output_dir
    )

    print("\n=== Running Reconstruction Attack on Fine-tuned Embeddings ===")
    finetuned_results = run_reconstruction_attack(
        finetuned_test_embeddings, test_df['sequence'].tolist(),
        "finetuned", output_dir
    )

    # Compare attacks
    delta_privacy, pretrained_acc, finetuned_acc = compare_reconstruction_attacks(
        pretrained_results, finetuned_results, output_dir
    )

    # Calculate privacy metrics
    privacy_gain = calculate_privacy_gain_log(pretrained_acc, finetuned_acc)
    avg_privacy_gain = np.mean(privacy_gain)
    avg_display_gain = (np.exp(avg_privacy_gain) - 1) * 100
    avg_display_gain = min(avg_display_gain, 50.0)
    print(f"Average Error-Based Privacy Gain: {avg_display_gain:.2f}%")

    # Calculate PDS metrics
    pds_delta = calculate_pds_delta(delta_privacy, utility_delta)
    pds_gain = calculate_pds_gain(privacy_gain, utility_delta)
    
    # Create PDS comparison
    disagreement_positions = create_pds_comparison(
        pds_delta, pds_gain, delta_privacy, privacy_gain, output_dir
    )
    
    # Calculate WWP
    wwp_delta = calculate_wwp(pds_delta)
    wwp_gain = calculate_wwp(pds_gain)
    
    print(f"Win-Win Percentage (Delta-Privacy): {wwp_delta:.2f}%")
    print(f"Win-Win Percentage (Privacy-Gain): {wwp_gain:.2f}%")
    print(f"Average PDS (Delta-Privacy): {np.mean(pds_delta):.2f}")
    print(f"Average PDS (Privacy-Gain): {np.mean(pds_gain):.2f}")

    # Statistical testing
    t_stat, p_value, mean_pg, se_pg = statistical_significance_test(privacy_gain)
    print(f"Privacy Gain t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    significance = "significant" if p_value < 0.05 else "not significant"
    print(f"Privacy improvement is statistically {significance}")

    # Visualize privacy metrics
    privacy_metrics = visualize_privacy_metrics(
        delta_privacy, 
        privacy_gain, 
        pds_delta,
        pds_gain,
        utility_delta,
        output_dir
    )

    # Create Pareto frontiers
    position_analysis = create_position_pareto_frontiers(
        delta_privacy, privacy_gain, utility_delta, output_dir
    )
    
    print(f"Win-Win positions (Delta-Privacy): {position_analysis['delta_privacy']['win_win_positions']}")
    print(f"Win-Win positions (Privacy-Gain): {position_analysis['privacy_gain']['win_win_positions']}")
    print(f"Common Win-Win positions: {position_analysis['common_win_win']}")
    
    # Final summary
    print("\n==== Results Summary ====")
    print(f"Model: XLNet-Large")
    print(f"Dataset: HS3D (Splice Site Prediction)")
    
    print("\nUtility Metrics:")
    print(f"  Pretrained Accuracy:  {pretrained_utility_metrics['accuracy']:.4f}")
    print(f"  Fine-tuned Accuracy:  {finetuned_utility_metrics['accuracy']:.4f}")
    print(f"  Utility Delta:        {utility_delta:.4f} ({utility_delta*100:.2f}%)")
    
    print("\nPrivacy Metrics:")
    print(f"  Pretrained Reconstruction: {np.mean(pretrained_acc):.4f}")
    print(f"  Fine-tuned Reconstruction: {np.mean(finetuned_acc):.4f}")
    print(f"  Delta Privacy:             {np.mean(delta_privacy):.4f}")
    print(f"  Privacy Gain:              {avg_display_gain:.2f}%")
    
    print("\nJoint Metrics (Delta-Privacy):")
    print(f"  PDS: {np.mean(pds_delta):.2f}")
    print(f"  WWP: {wwp_delta:.2f}%")
    
    print("\nJoint Metrics (Privacy-Gain):")
    print(f"  PDS: {np.mean(pds_gain):.2f}")
    print(f"  WWP: {wwp_gain:.2f}%")
    print(f"  Statistical: t={t_stat:.2f}, p={p_value:.4f} ({significance})")
    
    # Create final summary table
    summary_data = {
        'Metric': [
            'Pretrained Accuracy', 'Fine-tuned Accuracy', 'Utility Gain',
            'Pretrained Reconstruction', 'Fine-tuned Reconstruction',
            'Delta Privacy', 'Privacy Gain',
            'WWP (Delta)', 'WWP (Gain)',
            'PDS (Delta)', 'PDS (Gain)',
            't-statistic', 'p-value'
        ],
        'Value': [
            f"{pretrained_utility_metrics['accuracy']:.4f}",
            f"{finetuned_utility_metrics['accuracy']:.4f}",
            f"{utility_delta*100:.2f}%",
            f"{np.mean(pretrained_acc):.4f}",
            f"{np.mean(finetuned_acc):.4f}",
            f"{np.mean(delta_privacy)*100:.2f}%",
            f"{avg_display_gain:.2f}%",
            f"{wwp_delta:.2f}%",
            f"{wwp_gain:.2f}%",
            f"{np.mean(pds_delta):.2f}",
            f"{np.mean(pds_gain):.2f}",
            f"{t_stat:.4f}",
            f"{p_value:.6f}"
        ],
        'Interpretation': [
            "Baseline performance",
            "Fine-tuned performance",
            "Utility improvement",
            "Baseline vulnerability",
            "Fine-tuned vulnerability",
            "Accuracy change",
            "Error increase",
            "Win-win % (delta)",
            "Win-win % (gain)",
            "Balance (delta)",
            "Balance (gain)",
            "Significance measure",
            f"{significance}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}final_results_summary.csv", index=False)
    
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    
    table = plt.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc='center',
        cellLoc='left'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    for i in range(len(summary_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    
    is_win_win_delta = utility_delta > 0 and np.mean(delta_privacy) > 0
    is_win_win_gain = utility_delta > 0 and avg_privacy_gain > 0
    
    if is_win_win_delta:
        for j in range(len(summary_df.columns)):
            cell = table[(8, j)]
            cell.set_facecolor('#E2EFDA')
            cell.set_text_props(fontweight='bold')
            
    if is_win_win_gain:
        for j in range(len(summary_df.columns)):
            cell = table[(9, j)]
            cell.set_facecolor('#E2EFDA')
            cell.set_text_props(fontweight='bold')
    
    plt.title('XLNet-Large Privacy-Utility Analysis: Summary Results', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}final_results_summary.png", dpi=300)
    plt.close()
    
    print("\nAll analysis completed. Results saved to:", output_dir)
    
    return {
        "utility_metrics": {
            "pretrained": pretrained_utility_metrics,
            "finetuned": finetuned_utility_metrics,
            "delta": utility_delta
        },
        "privacy_metrics": {
            "pretrained_acc": np.mean(pretrained_acc),
            "finetuned_acc": np.mean(finetuned_acc),
            "delta_privacy": np.mean(delta_privacy),
            "privacy_gain": avg_display_gain,
            "privacy_significance": {
                "t_stat": t_stat,
                "p_value": p_value
            }
        },
        "joint_metrics_delta": {
            "wwp": wwp_delta,
            "avg_pds": np.mean(pds_delta),
            "is_win_win": is_win_win_delta
        },
        "joint_metrics_gain": {
            "wwp": wwp_gain,
            "avg_pds": np.mean(pds_gain),
            "is_win_win": is_win_win_gain
        },
        "common_win_win_positions": position_analysis['common_win_win']
    }

if __name__ == "__main__":
    main()