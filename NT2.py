# -*- coding: utf-8 -*-
"""
COMPLETE Nucleotide Transformer 2.5B Fine-tuning with IA3
Maintains ALL original functionality + memory optimizations
"""

import matplotlib
matplotlib.use('Agg')

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
import matplotlib.ticker as mtick
from matplotlib_venn import venn2
from peft import IA3Config, get_peft_model, prepare_model_for_kbit_training, TaskType
import gc

# ============================================================================
# REPRODUCIBILITY SEEDS
# ============================================================================
SEED = 42
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================================
# CONFIGURATION
# ============================================================================
train_path = 'hs3d_train_pan.csv'
test_path = 'hs3d_test_pan.csv'
model_name = "InstaDeepAI/nucleotide-transformer-2.5b-1000g"
output_dir = "./nucleotide-transformer-2.5b-result/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}attack_results/", exist_ok=True)

print("="*80)
print("NUCLEOTIDE TRANSFORMER 2.5B - COMPLETE PRIVACY-UTILITY ANALYSIS")
print("Method: IA3 (InstaDeepAI) + 4-bit Quantization + Gradient Checkpointing")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================================
def load_data():
    print("\n" + "="*80)
    print("Loading and preprocessing data...")
    print("="*80)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['sequence'] = train_df['sequence'].astype(str).str.upper()
    test_df['sequence'] = test_df['sequence'].astype(str).str.upper()
    
    train_df = train_df[~train_df['sequence'].str.contains('N', na=False)]
    test_df = test_df[~test_df['sequence'].str.contains('N', na=False)]

    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    print(f"Class distribution in training: {train_df['label'].value_counts().to_dict()}")
    print(f"Class distribution in testing: {test_df['label'].value_counts().to_dict()}")

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

# ============================================================================
# 2. CREATE DATASETS
# ============================================================================
def create_datasets(train_df, test_df):
    print("\n" + "="*80)
    print("Creating datasets and tokenizing...")
    print("="*80)
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def tokenize_function(examples):
        return tokenizer(
            examples['sequence'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

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

# ============================================================================
# 3. LOAD PRETRAINED MODEL (QUANTIZED)
# ============================================================================
def load_pretrained_model_quantized():
    print("\n" + "="*80)
    print("Loading pretrained model with 4-bit quantization...")
    print("="*80)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"? Pretrained model loaded (4-bit quantized)")
    return model

# ============================================================================
# 4. FINE-TUNE WITH IA3
# ============================================================================
def finetune_nucleotide_transformer_ia3(tokenized_train, tokenized_test, tokenizer):
    print("\n" + "="*80)
    print("FINE-TUNING WITH IA3 (InstaDeepAI's Method)")
    print("="*80)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    ia3_config = IA3Config(
        task_type=TaskType.SEQ_CLS,
        target_modules=["k_proj", "v_proj", "down_proj"],
        feedforward_modules=["down_proj"],
    )

    model = get_peft_model(model, ia3_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"? IA3 adapters applied")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
    print(f"  Total: {total_params:,}")

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        try:
            auc = roc_auc_score(labels, pred.predictions[:, 1])
        except:
            auc = 0.0
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

    training_args = TrainingArguments(
        output_dir=f"{output_dir}finetune_ia3",
        learning_rate=3e-3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        eval_accumulation_steps=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        torch_empty_cache_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    print("\nStarting training...")
    trainer.train()
    eval_result = trainer.evaluate()
    print(f"? Evaluation: {eval_result}")

    model.save_pretrained(f"{output_dir}finetune_ia3_adapters")
    tokenizer.save_pretrained(f"{output_dir}finetune_ia3_adapters")
    
    return model, eval_result

# ============================================================================
# 5. EXTRACT EMBEDDINGS
# ============================================================================
def extract_embeddings(model, tokenizer, sequences):
    print(f"Extracting CLS embeddings...")

    if hasattr(model, 'base_model'):
        if hasattr(model.base_model, 'model'):
            feature_extractor = model.base_model.model
        else:
            feature_extractor = model.base_model
    elif hasattr(model, 'esm'):
        feature_extractor = model.esm
    elif hasattr(model, 'bert'):
        feature_extractor = model.bert
    elif hasattr(model, 'model'):
        feature_extractor = model.model
    else:
        feature_extractor = model
    
    feature_extractor.eval()
    all_embeddings = []
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            inputs = tokenizer.batch_encode_plus(
                batch_sequences, padding='max_length', truncation=True,
                max_length=512, return_tensors='pt'
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            try:
                outputs = feature_extractor(**inputs, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states'):
                    cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                elif hasattr(outputs, 'last_hidden_state'):
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    cls_embeddings = outputs[0][:, 0, :].cpu().numpy()
            except:
                outputs = feature_extractor(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    cls_embeddings = outputs[0][:, 0, :].cpu().numpy()

            all_embeddings.append(cls_embeddings)
            if i % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    embeddings = np.vstack(all_embeddings)
    print(f"? Embeddings shape: {embeddings.shape}")
    return embeddings

# ============================================================================
# 6. MLP CLASSIFIER
# ============================================================================
class MLPClassifier(nn.Module):
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
    print(f"\nTraining MLP on {model_type} embeddings...")

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

    model = MLPClassifier(input_dim=train_embeddings.shape[1])
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
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

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
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
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_probs)

        all_epoch_metrics.append({
            'epoch': epoch + 1, 'loss': epoch_loss, 'accuracy': accuracy,
            'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc
        })

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {accuracy:.4f}, "
              f"Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    torch.save(model.state_dict(), f"{output_dir}mlp_classifier_{model_type}.pt")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_type}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}confusion_matrix_{model_type}.png")
    plt.close()

    return model, all_epoch_metrics

# ============================================================================
# 7. COMPREHENSIVE VISUALIZATIONS (ALL FROM ORIGINAL)
# ============================================================================
def create_comprehensive_visualizations(all_metrics):
    epochs = [m['epoch'] for m in all_metrics]
    loss = [m['loss'] for m in all_metrics]
    accuracy = [m['accuracy'] for m in all_metrics]
    precision = [m['precision'] for m in all_metrics]
    recall = [m['recall'] for m in all_metrics]
    f1 = [m['f1'] for m in all_metrics]
    auc = [m['auc'] for m in all_metrics]

    avg_loss = np.mean(loss)
    avg_accuracy = np.mean(accuracy)
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    avg_auc = np.mean(auc)

    # Training metrics
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracy, 'o-', linewidth=2, label='Accuracy')
    plt.plot(epochs, precision, 'o-', linewidth=2, label='Precision')
    plt.plot(epochs, recall, 'o-', linewidth=2, label='Recall')
    plt.plot(epochs, f1, 'o-', linewidth=2, label='F1 Score')
    plt.plot(epochs, auc, 'o-', linewidth=2, label='AUC')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Training Metrics by Epoch', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}training_metrics.png", dpi=300)
    plt.close()

    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'o-', color='crimson', linewidth=2)
    plt.axhline(y=avg_loss, color='crimson', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss by Epoch', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}loss_curve.png", dpi=300)
    plt.close()

    return {"accuracy": avg_accuracy, "precision": avg_precision, "recall": avg_recall,
            "f1": avg_f1, "auc": avg_auc}

def visualize_embeddings_pca(embeddings, labels, dataset_name):
    pca = PCA(n_components=2)
    sample_size = min(2000, len(embeddings))
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    embeddings_2d = pca.fit_transform(embeddings[indices])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=labels[indices], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title(f'PCA - {dataset_name}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}pca_{dataset_name.lower()}.png", dpi=300)
    plt.close()

# ============================================================================
# 8. RECONSTRUCTION ATTACK (COMPLETE)
# ============================================================================
class ReconstructionAttackModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=200):
        super(ReconstructionAttackModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.bn1(self.sigmoid(self.fc1(x)))
        x = self.bn2(self.sigmoid(self.fc2(x)))
        return self.fc3(x)

def create_positional_embedding(position, dim):
    pos_embedding = torch.zeros(dim)
    for i in range(0, dim, 2):
        pos_embedding[i] = np.sin(position / (10000 ** (i / dim)))
        if i + 1 < dim:
            pos_embedding[i + 1] = np.cos(position / (10000 ** ((i + 1) / dim)))
    return pos_embedding

def run_reconstruction_attack(embeddings, sequences, model_name, output_dir):
    print(f"\nRunning reconstruction attack on {model_name}...")
    
    embedding_dim = embeddings.shape[1]
    seq_length = 20
    valid_indices = [i for i, seq in enumerate(sequences) if len(seq) >= seq_length]
    embeddings = embeddings[valid_indices]
    sequences = [sequences[i] for i in valid_indices]

    nuc_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    position_accuracy, confusion_matrices = [], []

    for pos in range(seq_length):
        pos_embeddings = torch.stack([create_positional_embedding(pos, embedding_dim)
                                    for _ in range(len(embeddings))])
        y_true = torch.tensor([nuc_to_int.get(seq[pos].upper(), 0) for seq in sequences])
        X = torch.cat([torch.tensor(embeddings, dtype=torch.float32), pos_embeddings], dim=1)

        dataset = TensorDataset(X, y_true)
        train_size = int(0.8 * len(dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size])

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model = ReconstructionAttackModel(embedding_dim)
        if torch.cuda.is_available():
            model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(5):
            model.train()
            for batch_x, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, predicted = torch.max(model(batch_x), 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())

        acc = accuracy_score(all_true, all_preds)
        position_accuracy.append(acc)
        cm = confusion_matrix(all_true, all_preds, labels=[0, 1, 2, 3])
        confusion_matrices.append(cm)

    # Calculate class-wise accuracy
    class_accuracy = {}
    for nuc in ['A', 'C', 'G', 'T']:
        nuc_idx = nuc_to_int[nuc]
        accuracies = []
        for cm in confusion_matrices:
            if cm[nuc_idx].sum() > 0:
                accuracies.append(cm[nuc_idx, nuc_idx] / cm[nuc_idx].sum())
        if accuracies:
            class_accuracy[nuc] = np.mean(accuracies)

    # Plot
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(1, seq_length + 1)
    plt.plot(x_pos, position_accuracy, 'o-', linewidth=2, label=model_name)
    plt.axhline(y=np.mean(position_accuracy), color='red', linestyle='--',
                label=f'Avg: {np.mean(position_accuracy):.3f}')
    plt.axhline(y=0.25, color='gray', linestyle=':', label='Random')
    plt.title(f'Reconstruction Attack - {model_name}', fontsize=14)
    plt.xlabel('Position')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/reconstruction_accuracy_{model_name}.png", dpi=300)
    plt.close()

    return position_accuracy, confusion_matrices, class_accuracy

# ============================================================================
# 9. COMPARE ATTACKS
# ============================================================================
def compare_reconstruction_attacks(pretrained_results, finetuned_results, output_dir):
    pretrained_acc, finetuned_acc = pretrained_results[0], finetuned_results[0]

    plt.figure(figsize=(12, 7))
    x_pos = np.arange(1, len(pretrained_acc) + 1)
    plt.plot(x_pos, pretrained_acc, 'o-', linewidth=2, color='blue',
             label=f'Pretrained (Avg: {np.mean(pretrained_acc):.3f})')
    plt.plot(x_pos, finetuned_acc, 'o-', linewidth=2, color='red',
             label=f'Fine-tuned (Avg: {np.mean(finetuned_acc):.3f})')
    plt.axhline(y=0.25, color='gray', linestyle=':', label='Random')

    for i in range(len(pretrained_acc)):
        if finetuned_acc[i] < pretrained_acc[i]:
            plt.axvspan(i+0.5, i+1.5, alpha=0.2, color='green')

    plt.title('Reconstruction Attack Comparison', fontsize=14)
    plt.xlabel('Position')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/reconstruction_comparison.png", dpi=300)
    plt.close()

    delta_privacy = np.array(pretrained_acc) - np.array(finetuned_acc)
    return delta_privacy, pretrained_acc, finetuned_acc

# ============================================================================
# 10. PRIVACY GAIN CALCULATION
# ============================================================================
def calculate_privacy_gain_log(pretrained_acc, finetuned_acc):
    pretrained_error = 1 - np.array(pretrained_acc)
    finetuned_error = 1 - np.array(finetuned_acc)
    privacy_gain = np.zeros_like(pretrained_error)
    
    for i in range(len(pretrained_error)):
        if pretrained_error[i] > 0 and finetuned_error[i] > 0:
            ratio = min(finetuned_error[i] / pretrained_error[i], 1.5)
            privacy_gain[i] = np.log(ratio)
        elif finetuned_error[i] > pretrained_error[i]:
            privacy_gain[i] = np.log(1.5)
    
    return privacy_gain

# ============================================================================
# 11. PARETO DOMINANCE SCORES
# ============================================================================
def calculate_pds_delta(delta_privacy, utility_delta):
    pds = np.zeros(len(delta_privacy))
    for i in range(len(delta_privacy)):
        if delta_privacy[i] > 0 and utility_delta > 0:
            pds[i] = 2  # Win-Win
        elif delta_privacy[i] > 0 and utility_delta >= 0:
            pds[i] = 1  # Privacy Gain
        elif delta_privacy[i] == 0 and utility_delta == 0:
            pds[i] = 0  # Neutral
        elif delta_privacy[i] < 0 and utility_delta < 0:
            pds[i] = -2  # Lose-Lose
        else:
            pds[i] = -1  # Tradeoff
    return pds

def calculate_pds_gain(privacy_gain, utility_delta):
    pds = np.zeros(len(privacy_gain))
    for i in range(len(privacy_gain)):
        if privacy_gain[i] > 0.095 and utility_delta > 0:
            pds[i] = 2
        elif privacy_gain[i] > 0.095 and utility_delta == 0:
            pds[i] = 1
        elif privacy_gain[i] > 0.049 and utility_delta > 0:
            pds[i] = 1
        elif (0 <= privacy_gain[i] < 0.049 and utility_delta >= 0) or (privacy_gain[i] == 0 and utility_delta == 0):
            pds[i] = 0
        elif privacy_gain[i] < 0 and utility_delta < 0:
            pds[i] = -2
        else:
            pds[i] = -1
    return pds

def calculate_wwp(pds):
    return (np.sum(pds == 2) / len(pds)) * 100

def statistical_significance_test(privacy_gain):
    mean_pg = np.mean(privacy_gain)
    se_pg = stats.sem(privacy_gain)
    t_stat = mean_pg / se_pg if se_pg > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(privacy_gain)-1))
    return t_stat, p_value, mean_pg, se_pg

# ============================================================================
# 12. VISUALIZE PRIVACY METRICS (ALL FROM ORIGINAL)
# ============================================================================
def visualize_privacy_metrics(delta_privacy, privacy_gain, pds_delta, pds_gain, utility_delta, output_dir):
    seq_length = len(delta_privacy)
    x_pos = np.arange(1, seq_length + 1)
    
    # Privacy Gain
    plt.figure(figsize=(10, 6))
    display_values = np.minimum((np.exp(privacy_gain) - 1) * 100, 50.0)
    plt.bar(x_pos, display_values, color=['green' if x > 0 else 'red' for x in privacy_gain])
    plt.axhline(y=0, color='black', linestyle='-')
    avg_display = min((np.exp(np.mean(privacy_gain)) - 1) * 100, 50.0)
    plt.axhline(y=avg_display, color='blue', linestyle='--', label=f'Avg: {avg_display:.1f}%')
    plt.title('Error-Based Privacy Gain', fontsize=14)
    plt.xlabel('Position')
    plt.ylabel('Privacy Gain (%)')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/privacy_gain.png", dpi=300)
    plt.close()
    
    # PDS Distribution - Delta
    plt.figure(figsize=(10, 6))
    pds_counts = [(pds_delta == -2).sum(), (pds_delta == -1).sum(), 
                  (pds_delta == 0).sum(), (pds_delta == 1).sum(), (pds_delta == 2).sum()]
    pds_percentage = np.array(pds_counts) / len(pds_delta) * 100
    categories = ['Lose-Lose', 'Tradeoff', 'Neutral', 'Privacy Gain', 'Win-Win']
    colors = ['red', 'lightcoral', 'gray', 'lightgreen', 'green']
    
    bars = plt.bar(categories, pds_percentage, color=colors)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{pds_percentage[i]:.1f}%', ha='center', va='bottom')
    
    plt.title('PDS Distribution (Delta-Privacy)', fontsize=14)
    plt.ylabel('Percentage (%)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/pds_distribution_delta.png", dpi=300)
    plt.close()
    
    wwp_delta = calculate_wwp(pds_delta)
    wwp_gain = calculate_wwp(pds_gain)
    t_stat, p_value, _, _ = statistical_significance_test(privacy_gain)
    
    return {
        "win_win_percentage_delta": wwp_delta,
        "win_win_percentage_gain": wwp_gain,
        "avg_privacy_gain": avg_display,
        "avg_pds_delta": np.mean(pds_delta),
        "avg_pds_gain": np.mean(pds_gain),
        "t_statistic": t_stat,
        "p_value": p_value
    }

# ============================================================================
# 13. PARETO FRONTIERS (ALL FROM ORIGINAL)
# ============================================================================
def create_position_pareto_frontiers(delta_privacy, privacy_gain, utility_delta, output_dir):
    seq_length = len(delta_privacy)
    positions = np.arange(1, seq_length + 1)
    
    # Delta Privacy Frontier
    plt.figure(figsize=(12, 8))
    win_win_indices = np.where((delta_privacy > 0) & (utility_delta > 0))[0]
    privacy_only_indices = np.where((delta_privacy > 0) & (utility_delta <= 0))[0]
    
    plt.scatter(np.full(len(win_win_indices), utility_delta*100), 
               delta_privacy[win_win_indices]*100, s=100, c='green', alpha=0.7, label='Win-Win')
    plt.scatter(np.full(len(privacy_only_indices), utility_delta*100), 
               delta_privacy[privacy_only_indices]*100, s=100, c='orange', alpha=0.7, label='Privacy Only')
    
    for i, pos in enumerate(positions):
        plt.annotate(str(pos), (utility_delta*100, delta_privacy[i]*100),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    plt.title('Position-Specific Analysis: Delta Privacy', fontsize=14)
    plt.xlabel('Utility Gain (%)')
    plt.ylabel('Delta Privacy (%)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}attack_results/position_pareto_frontier_delta.png", dpi=300)
    plt.close()
    
    return {"delta_privacy": {"win_win_positions": [int(positions[i]) for i in win_win_indices]}}

# ============================================================================
# 14. MAIN EXECUTION
# ============================================================================
def main():
    print("\nSTARTING COMPLETE PRIVACY-UTILITY ANALYSIS WITH IA3\n")

    # Load data
    train_df, test_df = load_data()
    tokenized_train, tokenized_test, tokenizer = create_datasets(train_df, test_df)

    # PRETRAINED
    print("\n" + "="*80)
    print("PHASE 1: PRETRAINED MODEL")
    print("="*80)
    pretrained_model = load_pretrained_model_quantized()
    pretrained_train_emb = extract_embeddings(pretrained_model, tokenizer, train_df['sequence'].tolist())
    pretrained_test_emb = extract_embeddings(pretrained_model, tokenizer, test_df['sequence'].tolist())
    
    np.save(f"{output_dir}pretrained_train_embeddings.npy", pretrained_train_emb)
    np.save(f"{output_dir}pretrained_test_embeddings.npy", pretrained_test_emb)
    
    _, pretrained_metrics = train_mlp(pretrained_train_emb, train_df['label'].values,
                                       pretrained_test_emb, test_df['label'].values, "pretrained_ia3")
    pretrained_utility = create_comprehensive_visualizations(pretrained_metrics)
    visualize_embeddings_pca(pretrained_test_emb, test_df['label'].values, "Pretrained")
    
    del pretrained_model
    torch.cuda.empty_cache()
    gc.collect()

    # FINE-TUNED
    print("\n" + "="*80)
    print("PHASE 2: FINE-TUNING WITH IA3")
    print("="*80)
    finetuned_model, _ = finetune_nucleotide_transformer_ia3(tokenized_train, tokenized_test, tokenizer)
    finetuned_train_emb = extract_embeddings(finetuned_model, tokenizer, train_df['sequence'].tolist())
    finetuned_test_emb = extract_embeddings(finetuned_model, tokenizer, test_df['sequence'].tolist())
    
    np.save(f"{output_dir}finetuned_train_embeddings.npy", finetuned_train_emb)
    np.save(f"{output_dir}finetuned_test_embeddings.npy", finetuned_test_emb)
    
    _, finetuned_metrics = train_mlp(finetuned_train_emb, train_df['label'].values,
                                      finetuned_test_emb, test_df['label'].values, "finetuned_ia3")
    finetuned_utility = create_comprehensive_visualizations(finetuned_metrics)
    visualize_embeddings_pca(finetuned_test_emb, test_df['label'].values, "Finetuned")
    
    utility_delta = finetuned_utility["accuracy"] - pretrained_utility["accuracy"]
    print(f"\n? Utility delta: {utility_delta:.4f} ({utility_delta*100:.2f}%)")

    # RECONSTRUCTION ATTACKS
    print("\n" + "="*80)
    print("PHASE 3: PRIVACY ANALYSIS")
    print("="*80)
    pretrained_results = run_reconstruction_attack(pretrained_test_emb, test_df['sequence'].tolist(),
                                                     "pretrained", output_dir)
    finetuned_results = run_reconstruction_attack(finetuned_test_emb, test_df['sequence'].tolist(),
                                                    "finetuned", output_dir)
    
    delta_privacy, pretrained_acc, finetuned_acc = compare_reconstruction_attacks(
        pretrained_results, finetuned_results, output_dir)
    
    # PRIVACY METRICS
    privacy_gain = calculate_privacy_gain_log(pretrained_acc, finetuned_acc)
    pds_delta = calculate_pds_delta(delta_privacy, utility_delta)
    pds_gain = calculate_pds_gain(privacy_gain, utility_delta)
    
    privacy_metrics = visualize_privacy_metrics(delta_privacy, privacy_gain, pds_delta,
                                                 pds_gain, utility_delta, output_dir)
    position_analysis = create_position_pareto_frontiers(delta_privacy, privacy_gain,
                                                           utility_delta, output_dir)

    # FINAL SUMMARY
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"Method: IA3 (InstaDeepAI)")
    print(f"\nUtility:")
    print(f"  Pretrained: {pretrained_utility['accuracy']:.4f}")
    print(f"  Fine-tuned: {finetuned_utility['accuracy']:.4f}")
    print(f"  Gain: {utility_delta*100:.2f}%")
    print(f"\nPrivacy:")
    print(f"  Pretrained Attack: {np.mean(pretrained_acc):.4f}")
    print(f"  Fine-tuned Attack: {np.mean(finetuned_acc):.4f}")
    print(f"  Privacy Gain: {privacy_metrics['avg_privacy_gain']:.2f}%")
    print(f"\nJoint Metrics:")
    print(f"  WWP (Delta): {privacy_metrics['win_win_percentage_delta']:.2f}%")
    print(f"  WWP (Gain): {privacy_metrics['win_win_percentage_gain']:.2f}%")
    print(f"  PDS (Delta): {privacy_metrics['avg_pds_delta']:.2f}")
    print(f"  PDS (Gain): {privacy_metrics['avg_pds_gain']:.2f}")
    print(f"  t-statistic: {privacy_metrics['t_statistic']:.2f}")
    print(f"  p-value: {privacy_metrics['p_value']:.4f}")
    
    summary_df = pd.DataFrame({
        'Metric': ['Pretrained Acc', 'Fine-tuned Acc', 'Utility Gain (%)',
                   'Pretrained Attack', 'Fine-tuned Attack', 'Privacy Gain (%)',
                   'WWP-Delta (%)', 'WWP-Gain (%)', 'PDS-Delta', 'PDS-Gain'],
        'Value': [f"{pretrained_utility['accuracy']:.4f}",
                  f"{finetuned_utility['accuracy']:.4f}",
                  f"{utility_delta*100:.2f}",
                  f"{np.mean(pretrained_acc):.4f}",
                  f"{np.mean(finetuned_acc):.4f}",
                  f"{privacy_metrics['avg_privacy_gain']:.2f}",
                  f"{privacy_metrics['win_win_percentage_delta']:.2f}",
                  f"{privacy_metrics['win_win_percentage_gain']:.2f}",
                  f"{privacy_metrics['avg_pds_delta']:.2f}",
                  f"{privacy_metrics['avg_pds_gain']:.2f}"]
    })
    summary_df.to_csv(f"{output_dir}final_results_summary.csv", index=False)
    print(f"\n? Complete results saved to {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()