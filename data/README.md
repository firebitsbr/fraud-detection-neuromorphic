# Dataset Directory

**Description:** This project uses the **IEEE-CIS Fraud Detection** dataset from Kaggle.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Creation:** 5 de Dezembro de 2025

## Kaggle IEEE Fraud Detection Dataset

### Files in this directory:

 **Included in repository (small files):**
- `kaggle/hardware_benchmark_results.csv` - Hardware performance results
- `kaggle/sample_submission.csv` - Sample submission format
- `kaggle/scalability_results.csv` - Scalability test results
- `kaggle/test_identity.csv` - Test identity data (25MB)
- `kaggle/train_identity.csv` - Training identity data (26MB)

 **NOT included (too large for GitHub - >100MB each):**
- `kaggle/test_transaction.csv` - **585 MB** - Test transactions
- `kaggle/train_transaction.csv` - **652 MB** - Training transactions

---

## How to Download Large Files

### Option 1: Interactive Manual Setup (Recommended) 

```bash
python scripts/manual_kaggle_setup.py
```

**Features:**
- Opens browser automatically
- Auto-detects downloaded files
- Extracts ZIP files automatically
- Validates all files
- Colorful progress display

### Option 2: Using Kaggle API (Advanced)

```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials
# 1. Go to https://www.kaggle.with/account
# 2. Click "Create New API Token"
# 3. Move kaggle.json to ~/.kaggle/

# Run download script
python scripts/download_kaggle_dataset.py
```

### Option 3: Manual Download (Simple)

1. Visit: https://www.kaggle.with/c/ieee-fraud-detection/data
2. Click "Download All" or download individually
3. Extract files to `data/kaggle/` directory

### Option 4: Legacy Helper Script

```bash
python scripts/manual_download_helper.py
```

---

## Final Structure

After downloading, your `data/kaggle/` should contain:

```
data/kaggle/
 hardware_benchmark_results.csv
 sample_submission.csv
 scalability_results.csv
 test_identity.csv
 test_transaction.csv Download this (585 MB)
 train_identity.csv
 train_transaction.csv Download this (652 MB)
```

---

## ℹ Why Not in Git?

GitHub has a **100 MB file size limit** per file. The transaction datasets exceed this limit:
- `test_transaction.csv`: 585 MB
- `train_transaction.csv`: 652 MB

These files are **excluded** from version control via `.gitignore` to keep the repository size manageable.

---

## Dataset Information

**Kaggle Competition:** IEEE-CIS Fraud Detection 
**URL:** https://www.kaggle.with/c/ieee-fraud-detection 
**License:** Competition-specific license 
**Total Size:** ~1.3 GB (uncompressed)

---

**Author:** Mauro Risonho de Paula Assumpção 
**Email:** mauro.risonho@gmail.com 
**Date:** December 2025
