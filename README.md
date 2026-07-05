# CaCOM-HeartWISE

# Dataset

https://www.physionet.org/content/circor-heart-sound/1.0.3/

# General Timeline

## Exploratory Data Analysis (EDA)
- Metadata distributions
- Class balance
- Recording lengths
- Demographics
## Signal Analysis
- Waveforms
- Spectrograms
- Frequency content
- Noise inspection
## Feature Extraction
- MFCCs
- Chroma (optional)
- Spectral features
- Heart-rate and cycle features from the .tsv annotations
## Baseline Models (MFCCs, Spectral centroid, Bandwidth, Zero-crossing rate, RMS energy)
- Logistic Regression
- Random Forest
- XGBoost
## Deep Learning (log-Mel spectrograms)
- CNNs on log-Mel spectrograms
- Pretrained audio encoders (e.g., AST, Audio Spectrogram Transformer, or M2D)
- Transformer-based models for heart sound classification

---

# Signal Statistics (`signal_statistics.csv`)

The `signal_statistics.csv` file contains one row for every heart sound recording (`.wav`) in the CirCor DigiScope dataset. Each row summarizes several acoustic properties of the recording that are useful for exploratory data analysis, quality assessment, and machine learning feature engineering.

| Column | Description | Why it is useful |
|---------|-------------|------------------|
| **file** | Name of the audio file. | Identifies the recording. |
| **duration** | Length of the recording in seconds. | Detects unusually short or long recordings and helps determine if padding or cropping is required before training. |
| **samplerate** | Audio sampling frequency (Hz). | Ensures recordings were captured with a consistent sampling rate. CirCor recordings are typically sampled at **4000 Hz**. |
| **rms** | Root Mean Square (RMS) energy of the signal. Measures the average signal power (loudness). | Very small values may indicate silence or poor-quality recordings, while larger values indicate stronger heart sounds. |
| **peak** | Maximum absolute amplitude of the recording. | Used to detect clipping or recordings with unusually high amplitudes. |
| **dynamic_range** | Ratio between the peak amplitude and the RMS energy. | Indicates how much variation exists between quiet and loud parts of the recording. Higher values suggest more pronounced heart sounds or transient events. |
| **zcr** | Zero Crossing Rate (ZCR): the average number of times the waveform crosses zero amplitude. | Heart sounds typically have a low ZCR, whereas noisy recordings often have higher values. Useful for identifying recordings contaminated with high-frequency noise. |
| **spectral_centroid** | The "center of mass" of the frequency spectrum (Hz). | Indicates where most of the signal's energy is concentrated. Murmurs and noisy recordings generally exhibit higher spectral centroids than normal heart sounds. |
| **bandwidth** | Spectral bandwidth (Hz). Measures the spread of frequencies around the spectral centroid. | Larger bandwidths suggest broader frequency content, which may result from murmurs or background noise. |
| **rolloff** | Spectral rolloff frequency (Hz). The frequency below which approximately 85% of the signal energy is contained. | Helps distinguish low-frequency heart sounds from recordings containing significant high-frequency components. |
| **mfcc_mean** | Mean value of the first 13 Mel-Frequency Cepstral Coefficients (MFCCs). | Provides a compact representation of the recording's spectral characteristics and is commonly used in audio classification tasks. |
| **mfcc_std** | Standard deviation of the MFCCs. | Measures how much the spectral characteristics vary throughout the recording. |
| **silent** | Boolean flag indicating whether the recording has very low RMS energy (`rms < 0.005`). | Identifies potentially silent or unusable recordings. |
| **clipped** | Boolean flag indicating whether the recording amplitude reaches the maximum range (`peak > 0.99`). | Detects recordings that may suffer from clipping distortion due to saturation during recording. |

---

## Feature Categories

The extracted features can be grouped into four categories:

### Recording Metadata

- `file`
- `duration`
- `samplerate`

These describe the basic properties of each recording.

---

### Amplitude Features

- `rms`
- `peak`
- `dynamic_range`

These characterize the overall loudness and amplitude distribution of the signal.

---

### Frequency Features

- `spectral_centroid`
- `bandwidth`
- `rolloff`
- `zcr`

These summarize the frequency content of the recording and are useful for distinguishing normal heart sounds from murmurs or noisy recordings.

---

### Learned Audio Features

- `mfcc_mean`
- `mfcc_std`

Mel-Frequency Cepstral Coefficients (MFCCs) are widely used in speech and biomedical audio analysis because they provide a compact representation of the spectral envelope of a sound.

---

## Quality Control Flags

The following columns help identify recordings that may require exclusion or additional preprocessing:

- **silent**: Indicates recordings with extremely low signal energy.
- **clipped**: Indicates recordings that may contain clipping distortion.

These flags should be reviewed before training machine learning models to avoid introducing poor-quality data into the dataset.