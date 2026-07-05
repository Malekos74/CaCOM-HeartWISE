# CaCOM-HeartWISE

# Dataset

https://www.physionet.org/content/circor-heart-sound/1.0.3/


# Exploratory Data Analysis (EDA)
- Metadata distributions
- Class balance
- Recording lengths
- Demographics
# Signal Analysis
- Waveforms
- Spectrograms
- Frequency content
- Noise inspection
# Feature Extraction
- MFCCs
- Chroma (optional)
- Spectral features
- Heart-rate and cycle features from the .tsv annotations
# Baseline Models
- Logistic Regression
- Random Forest
- XGBoost
# Deep Learning
- CNNs on log-Mel spectrograms
- Pretrained audio encoders (e.g., AST, Audio Spectrogram Transformer, or M2D)
- Transformer-based models for heart sound classification