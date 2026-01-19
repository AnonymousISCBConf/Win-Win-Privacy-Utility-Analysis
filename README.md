## Repository Overview

The repository contains complete implementations for evaluating privacy-utility tradeoffs across 13 transformer architectures (9 general-purpose and 4 genomic foundation models) on splice site prediction tasks using the HS3D dataset. Each model implementation is self-contained and can be executed independently.

#### Dataset Description: HS3D Dataset

The HS3D (Homo Sapiens Splice Sites Dataset) must be placed in the repository root directory:

- `hs3d_train_pan.csv` -- Training set (31,680 sequences)
- `hs3d_test_pan.csv` -- Test set (2,000 sequences)

**Dataset format:**
```
sequence,label
ACGTACGTACGTACGTACGT,1
TGCATGCATGCATGCATGCA,0
...
```

Each sequence is 20 nucleotides long. Labels: 0 = negative, 1 = positive (splice site classification).

### Repository Structure

The repository uses a flat directory structure where all Python scripts and dataset files reside in the same location:

```
Win-Win-Privacy-Utility-Analysis/
├── README.md                    # Documentation
├── LICENSE                      # Apache 2.0 License
├── requirements.txt             # Python dependencies (84 packages)
├── .gitignore                  # Git ignore patterns
│
├── hs3d_train_pan.csv          # Training dataset (31,680 sequences)
├── hs3d_test_pan.csv           # Test dataset (2,000 sequences)
│
├── BertBase.py                 # BERT-Base implementation (110M)
├── BertLarge.py                # BERT-Large implementation (340M)
├── RobertaBase.py              # RoBERTa-Base implementation (125M)
├── RobertaLarge.py             # RoBERTa-Large implementation (355M)
├── XLBase.py                   # XLNet-Base implementation (110M)
├── XLLarge.py                  # XLNet-Large implementation (340M)
├── GPT2S.py                    # GPT-2 Small implementation (117M)
├── GPT2M.py                    # GPT-2 Medium implementation (345M)
├── Ernie.py                    # ERNIE 2.0 implementation (110M)
├── DNABERT.py                  # DNABERT-6 implementation (86M)
├── DNABERT2.py                 # DNABERT-2 implementation (117M)
├── DNAGPT.py                   # DNAGPT implementation (0.1B)
└── NT.py                       # Nucleotide Transformer (500M)
└── FinalTstaistics.py #statistical hypothesis testing visualization script 
```

**Result folders** (created after execution for each model, not in repository):
```
bert-result-double-win/   
bert-large-result-double-win/ 
roberta-base-result-double-win/ 
roberta-large-result-double-win/ 
xlnet-base-result-double-win/ 
xlnet-large-result-double-win/ 
gpt2-small-result-double-win/ 
gpt2-medium-result-double-win/ 
ernie-2.0-result-double-win/ 
dnabert-6-result-double-win/ 
dnabert2-result-double-win/               
dnagpt-result-complete/ 
nucleotide-transformer-500m-result/ 
hypothesis_plots/ 
```

### System Requirements

#### Hardware Requirements

- **GPU**: CUDA-compatible GPU with ≥16 GB VRAM
  - Recommended: NVIDIA A100 (40GB/80GB)
  - Minimum: NVIDIA RTX 3090 (24GB)
- **RAM**: ≥32 GB system memory (64 GB recommended)
- **Storage**: ≥50 GB free space

#### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+) or Windows 10/11
- **Python**: 3.10 (exact version)
- **CUDA**: 12.8 with compatible drivers
- **PyTorch**: 2.9.0 with CUDA 12.8 support

#### Key Dependencies

```
torch==2.9.0                # PyTorch with CUDA 12.8
transformers==4.57.1        # HuggingFace Transformers
datasets==4.2.0             # Dataset handling
numpy==2.2.6                # Numerical operations
pandas==2.3.3               # Data manipulation
scikit-learn==1.7.2         # ML utilities
matplotlib==3.10.7          # Visualization
seaborn==0.13.2             # Statistical plots
```

Complete dependency list (84 packages with exact versions) available in `requirements.txt`.

### Installation Instructions

#### Step 1: Clone Repository

```bash
git clone https://github.com/RecombConference/
Win-Win-Privacy-Utility-Analysis/tree/main
```

#### Step 2: Create Virtual Environment

```bash
python3.10 -m venv winwin_env
source winwin_env/bin/activate  # Linux/Mac
# Windows: winwin_env\Scripts\activate
```

#### Step 3: Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

#### Step 4: Install PyTorch with CUDA 12.8

```bash
pip install torch==2.9.0 --index-url \
    https://download.pytorch.org/whl/cu128
```

#### Step 5: Install All Dependencies

```bash
pip install -r requirements.txt
```

#### Step 6: Verify Installation

```bash
python -c "import torch; \
    print(f'PyTorch: {torch.__version__}'); \
    print(f'CUDA Available: {torch.cuda.is_available()}'); \
    print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.9.0+cu128
CUDA Available: True
CUDA Version: 12.8
```

### Execution Instructions

#### Running Individual Models

All model scripts are executed directly without command-line arguments:

```bash
# BERT-Base (fastest, recommended for testing)
python BertBase.py

# XLNet-Large (best win-win performance)
python XLLarge.py

# DNABERT-2 (genomic foundation model)
python DNABERT2.py
```

### Output Structure

Each model generates a dedicated result folder containing ~70 files:

```
bert-large-result-double-win/
│
├── attack_results/              # Reconstruction attack subfolder (~50 files)
│   ├── confusion_matrix_finetuned_pos_1.png through pos_20.png (20 files)
│   ├── confusion_matrix_pretrained_pos_1.png through pos_20.png (20 files)
│   ├── all_confusion_matrices_finetuned.png
│   ├── all_confusion_matrices_pretrained.png
│   ├── reconstruction_accuracy_finetuned.png
│   ├── reconstruction_accuracy_pretrained.png
│   ├── reconstruction_comparison.png
│   ├── nucleotide_accuracy_finetuned.png
│   ├── nucleotide_accuracy_pretrained.png
│   ├── nucleotide_comparison.png
│   ├── privacy_change.png
│   ├── privacy_gain.png
│   ├── privacy_utility_summary.png
│   ├── pareto_dominance_score_delta.png
│   ├── pareto_dominance_score_gain.png
│   ├── pds_distribution_delta.png
│   ├── pds_distribution_gain.png
│   ├── comparison_summary.xlsx
│   ├── reconstruction_results_finetuned.xlsx
│   └── reconstruction_results_pretrained.xlsx
│
├── finetuned_test_embeddings.npy
├── finetuned_train_embeddings.npy
├── pretrained_test_embeddings.npy
├── pretrained_train_embeddings.npy
│
├── mlp_classifier_finetuned_bert_large.pt
├── mlp_classifier_pretrained_bert_large.pt
│
├── class_distribution.png
├── sequence_length_distribution.png
├── confusion_matrix_finetuned_bert_large.png
├── confusion_matrix_pretrained_bert_large.png
├── learning_curves_finetuned_bert_large.png
├── learning_curves_pretrained_bert_large.png
├── loss_curve.png
├── training_metrics.png
├── precision_recall.png
├── f1_accuracy.png
├── average_metrics.png
├── metrics_table.png
├── tsne_finetuned_bert_large.png
├── tsne_pretrained_bert_large.png
├── pds_detailed_comparison.png
├── pds_heatmap_comparison.png
├── pds_comparison_summary.png
├── position_pareto_frontier_delta.png
├── position_pareto_frontier_gain.png
├── position_ranking_comparison.png
├── win_win_comparison.png
│
├── final_results_summary.png
└── final_results_summary.csv
```

**File count per model:** ~70 files, totaling 1.5--5.0 GB per model.


After running all 13 model scripts, execute the statistical hypothesis testing visualization:

```bash
python FinalTstatistics.py
```

This generates a separate `hypothesis_plots/` folder in the repository root:

```
hypothesis_plots/                   # Statistical testing results
│
├── hypothesis_test_BERT_base_privacy.png           (335 KB)
├── hypothesis_test_BERT_Large_privacy.png          (345 KB)
├── hypothesis_test_RoBERTa_base_privacy.png        (349 KB)
├── hypothesis_test_RoBERTa_Large_privacy.png       (352 KB)
├── hypothesis_test_XLNet_base_privacy.png          (345 KB)
├── hypothesis_test_XLNet_Large_privacy.png         (344 KB)
├── hypothesis_test_GPT2_Small_privacy.png          (342 KB)
├── hypothesis_test_GPT2_Medium_privacy.png         (347 KB)
├── hypothesis_test_ERNIE_20_privacy.png            (332 KB)
├── hypothesis_test_DNABERT_base_privacy.png        (350 KB)
├── hypothesis_test_DNABERT2_privacy.png            (348 KB)
├── hypothesis_test_DNAGPT_privacy.png              (343 KB)
├── hypothesis_test_Nucleotide_Transformer_privacy.png (348 KB)
└── hypothesis_tests_all_models_grid.png            (1,112 KB)
```

**File count:** 14 PNG files (13 individual + 1 grid), totaling ~4.7 MB.

Table: Script-to-Folder Output Mapping

| **Python Script** | **Generated Folder** |
|-------------------|---------------------|
| `BertBase.py` | `bert-result-double-win/` |
| `BertLarge.py` | `bert-large-result-double-win/` |
| `RobertaBase.py` | `roberta-base-result-double-win/` |
| `RobertaLarge.py` | `roberta-large-result-double-win/` |
| `XLBase.py` | `xlnet-base-result-double-win/` |
| `XLLarge.py` | `xlnet-large-result-double-win/` |
| `GPT2S.py` | `gpt2-small-result-double-win/` |
| `GPT2M.py` | `gpt2-medium-result-double-win/` |
| `Ernie.py` | `ernie-2.0-result-double-win/` |
| `DNABERT.py` | `dnabert-6-result-double-win/` |
| `DNABERT2.py` | `dnabert2-result-double-win/` |
| `DNAGPT.py` | `dnagpt-result-complete/` |
| `NT.py` | `nucleotide-transformer-500m-result/` |
| `FinalTstatistics.py` | `hypothesis_plots/` |

### Results Folder and Data Availability

The `results/` folder contains comprehensive experimental outputs from all 13 evaluated transformer architectures. Due to GitHub's file size constraints, the complete results have been hosted externally and are available via Google Drive[^1].

Running any model script locally generates the complete output (~72 files) as described in Section Installation Instructions.

The `hypothesis_plots` folder, which includes 14 PNG visualization files, is available online at Google Drive[^2].

[^1]: https://drive.google.com/drive/folders/12S5GZHYc00l28-7WV9X3E8J_XIqxrYdT?usp=sharing
[^2]: https://drive.google.com/drive/folders/1438k_MdtwebVJkV_VU4-JC1PUSC5SmgZ?usp=sharing

### Reproducibility

#### Fixed Random Seeds

All scripts use fixed seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

#### Expected Variability

Minor variations (±1--2% accuracy) are normal due to:
- Hardware differences (A100 vs V100 vs RTX)
- CUDA/PyTorch numerical precision
- Random initialization effects

Deviations >5% indicate setup issues.

## Quick Test Demo

**Recommended test:** Run BERTBase.py to verify installation:

```bash
python BertBase.py
```

**Expected output:** `bert-result-double-win/` folder with ~72 files (1.5--2.0 GB).

**Quick verification:** Script completes without errors, generates all expected files. Compare outputs with reference in `results/bert-result-double-win/` folder structure.

### Troubleshooting

#### Issue: Dataset Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 
'hs3d_train_pan.csv'
```

**Solution:** Ensure dataset files are in the repository root directory (same folder as Python scripts).

#### Issue: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. Edit script and reduce `per_device_train_batch_size` from 32 to 16 or 8
2. Use smaller models first (BERT-Base before BERT-Large)
3. Clear GPU cache: `torch.cuda.empty_cache()`

#### Issue: ImportError for Transformers

**Error:**
```
ImportError: cannot import name 'XXX' from 'transformers'
```

**Solution:**
```bash
pip install transformers==4.57.1 --force-reinstall
```

### License and Availability

- **License:** Apache 2.0
- **Repository:** [https://github.com/AnonymousISCBConf/Win-Win-Privacy-Utility-Analysis]
