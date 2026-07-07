# CaCOM-HeartWISE

Automated heart sound classification using machine learning on the CirCor DigiScope Phonocardiogram dataset.

The goal of this project is to predict **patient outcomes** from heart sound recordings by combining handcrafted audio feature engineering with classical machine learning and deep learning models.

**Dataset**: https://www.physionet.org/content/circor-heart-sound/1.0.3/

---

# Project Overview

Heart auscultation remains one of the primary diagnostic tools for detecting cardiovascular abnormalities. This project investigates automated heart sound classification using the CirCor DigiScope Phonocardiogram dataset.

The project explores multiple approaches ranging from classical signal processing and feature engineering to modern deep learning methods.

Current objectives include:

* Exploratory data analysis
* Signal analysis
* Audio feature extraction
* Classical machine learning baselines
* Deep learning on spectrograms
* Performance comparison and ablation studies

---

# Project Status

## Completed

* Exploratory dataset statistics
* Signal quality analysis
* Handcrafted feature extraction
* Patient-level train / validation / test splitting

* Logistic Regression baseline

## In Progress

* Random Forest baseline
* XGBoost baseline
* Hyperparameter tuning

## Planned

* CNN baseline
* Transformer-based models
* Feature importance analysis
* Model explainability
* Ensemble methods

---

# Repository Structure

> **TODO:** Add the complete repository tree.

## `data/`

Contains all raw and processed datasets.

Typical contents:

* `raw/training_data.csv`
* `processed/patient_splits.csv`
* `processed/feature_dataset.csv`

---
## `src/` 

### `features/`

Implements all signal processing and feature extraction code.

Responsibilities:

* Audio loading
* Signal preprocessing
* Time-domain features
* Frequency-domain features
* MFCC extraction
* Feature aggregation

---

### `models/`

Contains all machine learning models.

Current models:

* Logistic Regression

Planned:

* Random Forest
* XGBoost
* Support Vector Machine
* Neural Networks

---

## `notebooks/`

Exploratory notebooks used for experimentation and visualization.

---

## `scripts/`

> **TODO**

Document all standalone scripts and their purpose.

### `make_patient_splits.py`

Create the Train/Val/Test Splits using the features generated beforehand.

---

## `results/`

> **TODO**

Store trained models, evaluation metrics and generated figures.

---

# Dataset

## Source

CirCor DigiScope Phonocardiogram Dataset

## Labels

## Dataset Variables

The CirCor DigiScope dataset provides patient-level metadata, clinical annotations, and diagnostic labels.

| Variable | Description | Data Type | Possible Values |
|----------|-------------|-----------|-----------------|
| **Age** | Age category of the subject | String | Neonate, Infant, Child, Adolescent, Young adult |
| **Sex** | Reported sex of the subject | String | Female, Male |
| **Height** | Subject height in centimeters | Number | > 0 |
| **Weight** | Subject weight in kilograms | Number | > 0 |
| **Pregnancy status** | Whether the subject reported being pregnant at examination time | Boolean | True, False |
| **Additional ID** | Secondary identifier for subjects participating in both screening campaigns | String | Subject identifier |
| **Campaign** | Screening campaign attended by the subject | String | CC2014, CC2015 |
| **Murmur** | Presence of a murmur according to annotator | String | Present, Absent, Unknown |
| **Murmur locations** | Auscultation locations where at least one murmur was detected | String | Any combination of PV, TV, AV, MV, Phc separated by `+` |
| **Most audible location** | Auscultation location where the murmur was most intense | String | PV, TV, AV, MV, Phc |
| **Systolic murmur timing** | Timing of the murmur during systole | String | Early-systolic, Holosystolic, Late-systolic, Mid-systolic |
| **Systolic murmur shape** | Shape of the murmur during systole | String | Crescendo, Decrescendo, Diamond, Plateau |
| **Systolic murmur pitch** | Pitch of the systolic murmur | String | Low, Medium, High |
| **Systolic murmur grading** | Murmur intensity according to Levine's scale | String | I/VI, II/VI, III/VI |
| **Systolic murmur quality** | Acoustic quality of the systolic murmur | String | Blowing, Harsh, Musical |
| **Diastolic murmur timing** | Timing of the murmur during diastole | String | Early-diastolic, Holodiastolic, Mid-diastolic |
| **Diastolic murmur shape** | Shape of the murmur during diastole | String | Decrescendo, Plateau |
| **Diastolic murmur pitch** | Pitch of the diastolic murmur | String | Low, Medium, High |
| **Diastolic murmur grading** | Murmur intensity according to Levine's scale | String | I/IV, II/IV, III/IV |
| **Diastolic murmur quality** | Acoustic quality of the diastolic murmur | String | Blowing, Harsh |
| **Outcome** | Final diagnosis provided by expert cardiologist | String | Normal, Abnormal |

Describe:

* Outcome classes
* Murmur classes
* Number of patients
* Number of recordings
* Class distribution

---

## Patient Split

Patients are divided into:

* Training (70%)
* Validation (15%)
* Test (15%)

using a **patient-level stratified split**, ensuring recordings from the same patient never appear in multiple subsets.

---

# Machine Learning Pipeline

The project follows the pipeline below:

```
Raw WAV files
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Signal Analysis
        │
        ▼
Feature Extraction
        │
        ▼
feature_dataset.csv
        │
        ▼
Preprocessing
        │
        ▼
Model Training
        │
        ▼
Evaluation
```

---

# Exploratory Data Analysis (EDA)

Current analyses include:

* Class balance
* Recording duration distribution
* Patient demographics
* Recording locations
* Signal quality
* Noise inspection

> **TODO**

Add figures.

---

# Signal Analysis

Current analyses include:

* Raw waveforms
* FFT
* Frequency spectrum
* Mel spectrograms
* Noise inspection

> **TODO**

Add example visualizations.

---

# Signal Statistics (`signal_statistics.csv`)

Each row corresponds to one recording and summarizes global signal properties used for quality control.

## Available Features

| Feature           | Description             |
| ----------------- | ----------------------- |
| file              | Audio filename          |
| duration          | Recording length        |
| samplerate        | Sampling frequency      |
| rms               | Signal energy           |
| peak              | Maximum amplitude       |
| dynamic_range     | Peak-to-RMS ratio       |
| zcr               | Zero Crossing Rate      |
| spectral_centroid | Frequency center        |
| bandwidth         | Spectral bandwidth      |
| rolloff           | Spectral roll-off       |
| mfcc_mean         | Mean MFCC               |
| mfcc_std          | MFCC standard deviation |
| silent            | Silent recording flag   |
| clipped           | Clipping flag           |

---

## Quality Control

Current quality checks detect:

* Silent recordings
* Clipped recordings

---

# Feature Extraction

Each recording is converted into a fixed-length numerical feature vector.

Current feature groups include:

* Time-domain features
* Frequency-domain features
* Frequency band energy
* MFCC statistics
* MFCC delta features

---

## Time-Domain Features

| Feature       | Description         |
| ------------- | ------------------- |
| rms           | Signal energy       |
| peak          | Maximum amplitude   |
| variance      | Signal variance     |
| mean          | Signal mean         |
| std           | Standard deviation  |
| skew          | Waveform asymmetry  |
| kurtosis      | Waveform peakedness |
| zcr           | Zero Crossing Rate  |
| dynamic_range | Peak-to-RMS ratio   |

---

## Frequency-Domain Features

| Feature            | Description            |
| ------------------ | ---------------------- |
| spectral_centroid  | Center frequency       |
| spectral_bandwidth | Spectral spread        |
| spectral_rolloff   | Spectral boundary      |
| spectral_flatness  | Noise vs tonal content |

---

## Frequency Band Energy

| Feature             | Frequency Range |
| ------------------- | --------------- |
| band_energy_20_50   | 20–50 Hz        |
| band_energy_50_100  | 50–100 Hz       |
| band_energy_100_200 | 100–200 Hz      |
| band_energy_200_400 | 200–400 Hz      |
| band_energy_400_800 | 400–800 Hz      |

---

## MFCC Features

Current MFCC representation includes:

* Mean coefficients
* Standard deviation
* First-order deltas
* Second-order deltas

---

# Feature Dataset

Each row in `feature_dataset.csv` represents one recording.

The dataset currently contains:

* Patient metadata
* Recording information
* Train / validation / test split
* Audio features
* Outcome label

> **TODO**

Document every column in the dataset.

---

# Models

## Current Baselines

* Majority-class baseline
* Logistic Regression

## Planned Baselines

* Random Forest
* XGBoost
* Support Vector Machine

## Planned Deep Learning Models

* Multilayer Perceptron
* CNN on log-Mel spectrograms
* Audio Spectrogram Transformer (AST)
* Other pretrained audio foundation models

---

# Evaluation

Current evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score

> **TODO**

Include:

* Confusion matrix
* ROC curve
* Precision-Recall curve
* Cross-validation strategy

---

# How to Run

> **TODO**

Include commands for:

* Dataset preparation
* Patient splitting
* Feature extraction
* Model training
* Model evaluation

---

# Experiments

> **TODO**

Document all experiments.

Suggested sections:

* Metadata only
* Audio features only
* Combined features
* Feature ablation
* Hyperparameter search

---

# Results

> **TODO**

Summarize model performance.

Suggested table:

| Model | Validation Accuracy | Test Accuracy | Notes |
| ----- | ------------------- | ------------- | ----- |

---

# Future Work

Potential future directions include:

* Additional handcrafted features
* Heart-cycle segmentation
* Feature selection
* SHAP explainability
* Deep learning architectures
* Self-supervised audio models
* Ensemble learning
* Clinical interpretability

---

# References

> **TODO**

Add references for:

* CirCor Dataset
* Relevant papers
* Feature extraction methods
* Machine learning models
