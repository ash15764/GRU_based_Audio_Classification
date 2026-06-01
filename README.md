# Assignment #4 — Environmental Sound Classification
**Alexandria University | CSE: Pattern Recognition**
 
## Overview
GRU-based deep learning classifier for the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset — 8,732 labelled urban audio clips across 10 classes.
 
## Dataset Splits
| Split | Folds | Clips |
|-------|-------|-------|
| Train | 1–6 | ~5,435 |
| Validation | 7–8 | ~1,644 |
| Test | 9–10 | ~1,653 |
 
## Features (565 dims)
Log-Mel (128) · MFCCs + Δ + ΔΔ (120) · GFCCs (40) · Multi-scale Mel short/long (256) · Chroma (12) · Spectral Contrast (6) · RMS · ZCR · Rolloff
 
## Models
- **EnhancedGRUClassifier** — BiGRU + multi-head attention (4 heads) + residual skip
- **DeepCNNGRUClassifier** *(bonus)* — Residual CNN (3 blocks + SE attention) → BiGRU + multi-head attention
## Results
| Config | Model | Val Acc | Val F1 |
|--------|-------|---------|--------|
| **DCNNGRU-256-2-do30** | DeepCNNGRU | **0.7220** | **0.7362** |
| GRU-256-2-lr2e4 | GRU | 0.6989 | 0.7195 |
| GRU-256-2-do30 | GRU | 0.6953 | 0.7121 |
 
**Test set (best model):** Accuracy = **70.42%** · Macro F1 = **0.729**
 
## Requirements
```
pip install torch librosa gammatone numpy pandas matplotlib seaborn scikit-learn
```
 
## Usage
1. Download UrbanSound8K and place under `UrbanSound8K/`
2. Run `Classifier_Final.ipynb` top to bottom
3. Features are cached in `feature_cache_v3/` after the first run
