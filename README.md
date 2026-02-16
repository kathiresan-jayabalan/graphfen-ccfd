[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18618197.svg)](https://doi.org/10.5281/zenodo.18618197)

# GRAPHFEN-CCFD  
Graph-Based Heterogeneous Fraud Embedding Network for Credit Card Fraud Detection

This repository provides a **Jupyter Notebook** implementation of a Graph Neural Network-based pipeline (GRAPHFEN) for **credit card fraud detection and evaluation** using the public Kaggle `creditcard.csv` dataset.

**Main artifact:** `notebooks/graphfen-ccfd.ipynb`  
**End-to-end execution:** data loading → heterogeneous graph construction → embedding learning → classification → evaluation

---

## Overview

Credit card fraud detection is a highly imbalanced classification problem where **relational structure and entity interactions** play a crucial role.

This notebook implements a GRAPHFEN-style workflow that:

- Constructs a **heterogeneous graph** with multiple node types:
  - `card`
  - `merchant`
  - `device`
- Defines relations such as:
  - `card → merchant`
  - `card → device`
- Learns node embeddings using a **heterogeneous Graph Neural Network (GNN)** encoder.
- Forms a transaction representation by concatenating:
  - Card embedding  
  - Merchant embedding  
  - Device embedding  
  - Transaction feature vector  
- Performs supervised fraud classification using cross-entropy loss.

All stages of the experiment are implemented inside a single notebook for clarity and reproducibility.

---

## Methodology

The notebook performs the following steps:

- Load and preprocess the dataset `creditcard.csv`  
  - Feature normalization / scaling  
  - Optional feature selection  

- Generate synthetic entity identifiers (when real IDs are unavailable):
  - Hash-based bucketing from existing columns such as `Time`, binned `Amount`, and selected PCA features (e.g., V1–V4)
  - Create indices for `card`, `merchant`, and `device` nodes  

- Construct a heterogeneous graph using:
  - `PyTorch Geometric`
  - `HeteroData`
  - `HeteroConv`
  - `SAGEConv`

- Learn node embeddings via a hetero GNN encoder  

- Build transaction-level representations by concatenating involved node embeddings with transaction features  

- Train a supervised classifier using:
  - Cross-entropy loss for binary fraud classification (`Class ∈ {0,1}`)

- Track evaluation metrics including ROC AUC, PR AUC, confusion matrix, and classification scores.

---

## Dataset

The experiments use the public **Credit Card Fraud Detection** dataset (European cardholders, anonymized features) available on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (subject to Kaggle terms of use)

After downloading, place the file at:

data/creditcard.csv

If the full dataset is unavailable, a small synthetic or reduced subset may be used to verify execution. Such data will not reproduce reported performance metrics.

---

## Environment

Tested configuration:
- Python 3.x
- Jupyter Notebook or JupyterLab
- PyTorch
- GPU: NVIDIA T4 GPU (cloud or Colab)

---

## Dependencies

The notebook relies on the following Python packages. Install dependencies from `requirements.txt`

- numpy  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn  
- torch  
- torch-geometric  
- torch-scatter  
- torch-sparse  
- tqdm  
- ipywidgets  
- jupyterlab  
- pyyaml  

---

## How to Run

1. Download the dataset from Kaggle.  
2. Place `creditcard.csv` under the `data/` directory.  
3. Launch Jupyter Notebook or JupyterLab.  
4. Open `notebooks/graphfen-ccfd.ipynb`.  
5. Run all cells from top to bottom.

The notebook will preprocess data, construct the heterogeneous graph, train the GNN encoder and classifier, and evaluate the model.

Runtime depends on hardware and dataset size.

---

## Outputs

All evaluation metrics and visualizations are generated directly within the notebook output cells. No result files are written to disk by default.

The notebook prints final and per-epoch metrics at the end of training and evaluation.

- ROC AUC  
- PR AUC  
- Precision  
- Recall  
- F1-score  
- Accuracy  
- Confusion Matrix  

Plots such as loss curves, ROC curves, and precision-recall curves may be displayed inline during execution depending on enabled cells.

Example performance values observed during testing include strong recall for the minority (fraud) class and improved relational discrimination through graph embeddings. Exact results depend on graph construction parameters and training configuration.

- Accuracy ≈ 0.998x  
- Precision ≈ 0.6x  
- Recall ≈ 0.83x  
- F1 ≈ 0.7x  
- ROC AUC ≈ 0.94x  

---

## License

This project is licensed under the Apache License 2.0.

---

## Citation

If you use this repository in academic or research work, please cite:

Jayabalan, K. (2026). GraphFEN-CCFD: Heterogeneous GNN Model for Credit Card Fraud Detection (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.18618197
