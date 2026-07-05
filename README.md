# CaCOM-HeartWISE

A machine learning pipeline for **heart sound classification** using the PhysioNet CirCor DigiScope dataset.

Dataset:  
https://www.physionet.org/content/circor-heart-sound/1.0.3/

---

# 📌 Project Overview

This project aims to classify heart sound recordings into normal vs abnormal (murmur detection) using:

- Handcrafted audio features
- Classical machine learning models
- Deep learning models on spectrograms

The pipeline is structured in four stages:

---

# 🧭 Pipeline Roadmap

## 1. Exploratory Data Analysis (EDA)
Understanding dataset structure and quality:

- Class balance (Normal vs Abnormal / Murmur vs No Murmur)
- Recording duration distribution
- Patient demographics
- Recording location analysis
- Noise and quality inspection

---

## 2. Signal Analysis
Visual and spectral inspection of audio signals:

- Raw waveforms
- FFT / frequency content
- Mel spectrograms
- Noise patterns and artifacts

---

## 3. Feature Extraction
Conversion of audio into numerical representations:

### Feature groups:
- Time-domain features
- Frequency-domain features
- MFCC features (with deltas)
- Heart-cycle features (from `.tsv` annotations)

---

## 4. Modeling

### Classical ML (feature-based)
- Logistic Regression
- Random Forest
- XGBoost

### Deep Learning (spectrogram-based)
- CNNs on log-Mel spectrograms
- Pretrained audio models (AST, M2D, etc.)
- Transformer-based architectures

---

# 📊 Signal Statistics (`signal_statistics.csv`)

Each row represents one `.wav` recording with global signal properties used for EDA and quality control.

## Columns

| Feature | Description |
|--------|-------------|
| **file** | Audio file name |
| **duration** | Recording length (seconds) |
| **samplerate** | Sampling frequency (Hz, typically 4000 Hz) |
| **rms** | Average signal energy (loudness) |
| **peak** | Maximum amplitude |
| **dynamic_range** | Ratio of peak to RMS energy |
| **zcr** | Zero crossing rate (noise indicator) |
| **spectral_centroid** | Frequency “center of mass” |
| **bandwidth** | Spread of frequency content |
| **rolloff** | Frequency below which 85% energy is contained |
| **mfcc_mean** | Average MFCC value |
| **mfcc_std** | MFCC variability |
| **silent** | Low-energy recording flag |
| **clipped** | Signal saturation flag |

---

## 🧪 Quality Control Flags

- **silent** → potentially unusable recordings
- **clipped** → distorted recordings due to amplitude saturation

---

# ⚙️ Feature Extraction Overview

Each audio recording is transformed into a **fixed-length feature vector** consisting of:

- Time-domain statistics
- Frequency-domain descriptors
- Band-limited energy features
- MFCC-based representations

---

# ⏱️ 1. Time-Domain Features

These describe waveform shape and amplitude behavior.

| Feature | Description |
|--------|-------------|
| **rms** | Signal energy |
| **peak** | Maximum amplitude |
| **variance** | Signal spread |
| **mean** | Average amplitude |
| **std** | Standard deviation |
| **skew** | Asymmetry of waveform |
| **kurtosis** | Peakedness / outliers |
| **zcr** | Zero-crossing rate (noise indicator) |
| **dynamic_range** | Peak-to-RMS ratio |

---

# 🌊 2. Frequency-Domain Features

These describe how energy is distributed across frequencies.

| Feature | Description |
|--------|-------------|
| **spectral_centroid** | Frequency center of mass |
| **spectral_bandwidth** | Spread of spectrum |
| **spectral_rolloff** | Energy concentration boundary |
| **spectral_flatness** | Noise vs tonal structure |

---

## 📡 Frequency Band Energy Features

Energy in clinically meaningful frequency ranges:

| Feature | Frequency Range | Interpretation |
|--------|----------------|---------------|
| **band_energy_20_50** | 20–50 Hz | S1 / S2 fundamentals |
| **band_energy_50_100** | 50–100 Hz | Primary heart sound energy |
| **band_energy_100_200** | 100–200 Hz | Early murmur components |
| **band_energy_200_400** | 200–400 Hz | Murmur activity |
| **band_energy_400_800** | 400–800 Hz | Noise / harsh murmurs |

---

# 🎵 3. MFCC Features

MFCCs represent the spectral envelope of the signal.

For each coefficient (1–13):

### Base features
- **mfcc_i_mean**
- **mfcc_i_std**

### Temporal dynamics
- **mfcc_d1_i_mean / std** → first derivative (change over time)
- **mfcc_d2_i_mean / std** → second derivative (acceleration)

---

# 🧠 4. Feature Interpretation (Why this works)

Heart sounds consist of:

- S1 (lub)
- S2 (dub)
- Systole / diastole phases
- Possible murmurs (abnormal turbulence)

These features capture:

### Energy structure
- RMS, peak, dynamic range

### Frequency behavior
- Spectral centroid, rolloff, band energy

### Noise vs signal structure
- ZCR, spectral flatness

### Spectral shape (learned representation)
- MFCCs + deltas

---

# 📦 Dataset Format

Each row corresponds to one recording:

```text
[file, feature_1, feature_2, ..., feature_n, label]
```

---

# 🤖 Compatible Models

This feature set can be used with:

### Classical ML
- Logistic Regression
- Random Forest
- XGBoost
- SVM

### Neural Models
- MLP on tabular features
- CNN on spectrograms
- Transformer-based audio models

---

# 📌 Key Idea

This project combines:

- **Signal processing**
- **Clinical domain knowledge (heart cycles)**
- **Machine learning feature engineering**
- **Deep learning representations**

to build a robust heart sound classification system.
```