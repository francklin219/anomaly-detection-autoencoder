# Anomaly Detection on ECG Time Series with Autoencoder

This project implements an **unsupervised deep learning approach** for **anomaly detection on ECG time-series data** using an **Autoencoder neural network** built with **PyTorch**.

The model is trained exclusively on **normal heartbeats** and detects anomalies based on **reconstruction error**, a standard and effective technique in healthcare anomaly detection.

---

## Project Overview

- **Task**: Anomaly detection on ECG time series  
- **Approach**: Autoencoder (unsupervised deep learning)  
- **Dataset**: ECG5000  
- **Framework**: PyTorch  
- **Domain**: Healthcare AI / Time Series Analysis  

---

## Dataset

The **ECG5000** dataset contains electrocardiogram (ECG) signals labeled as:
- `0` → Normal heartbeat
- `1` → Anomalous heartbeat

Each sample consists of **140 time steps**.

The autoencoder is trained **only on normal samples**  
Anomalies are detected via reconstruction error at inference time

---

## Model Architecture

- Fully connected **Autoencoder**
- Encoder → Bottleneck → Decoder
- Loss function: **Mean Squared Error (MSE)**
- Anomaly score: **Reconstruction error**
- Threshold: **95th percentile of training reconstruction error**

---

## Processing Pipeline

1. Download ECG5000 dataset  
2. Preprocess & normalize signals  
3. Train autoencoder on normal data  
4. Compute reconstruction error  
5. Detect anomalies via thresholding  
6. Evaluate performance (AUC, F1-score, Confusion Matrix)

---

## Experimental Results

### Test Set Performance

- **AUC (reconstruction error)**: **0.9796**
- **F1-score**: **0.9516**
- **Threshold**: 95th percentile (training set)

### Confusion Matrix

[[2496 131]
[ 54 1819]]


These results show **strong anomaly detection performance** with a very low false negative rate, which is critical in healthcare applications.

---

## Project Structure

anomaly-detection-autoencoder/
│
├── data/
│ ├── raw/ # Raw ECG5000 files
│ └── processed/ # Preprocessed numpy arrays
│
├── models/
│ └── best_ae.pt # Best trained autoencoder
│
├── src/
│ ├── preprocess.py # Data preprocessing
│ ├── dataset.py # PyTorch Dataset class
│ └── models.py # Autoencoder architecture
│
├── train_autoencoder.py # Training & evaluation script
├── download_data.py # Dataset download script
├── README.md
└── .gitignore


---

## How to Run the Project

### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Download dataset:
python download_data.py
Preprocess data:
python -m src.preprocess
5️Train autoencoder:
python train_autoencoder.py
Why Autoencoder for Anomaly Detection?
No anomaly labels required during training

Robust to rare-event detection

Commonly used in:

Healthcare monitoring

Fraud detection

Industrial anomaly detection

Time-series surveillance systems

Future Improvements
LSTM / Conv1D Autoencoder

Variational Autoencoder (VAE)

Dynamic or adaptive thresholding

Explainability on reconstruction error

Deployment via API or Streamlit dashboard

Author:
Francklin
Machine Learning & Deep Learning Projects
Focus on Healthcare AI, Time-Series Modeling, and Anomaly Detection

