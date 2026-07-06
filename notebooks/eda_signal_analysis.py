from pathlib import Path
import random

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/training_data")
OUTPUT_DIR = Path("plots/eda/signal_analysis")

OUTPUT_DIR.mkdir(exist_ok=True)

wav_files = sorted(DATA_DIR.glob("*.wav"))

print(f"Found {len(wav_files)} recordings")

# ============================================================
# Containers
# ============================================================

rows = []

# ============================================================
# Analyze every recording
# ============================================================

for wav in tqdm(wav_files):

    signal, sr = librosa.load(wav, sr=None)

    duration = len(signal) / sr

    rms = np.sqrt(np.mean(signal**2))

    peak = np.max(np.abs(signal))

    dynamic_range = peak / (rms + 1e-8)

    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))

    centroid = np.mean(librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    ))

    bandwidth = np.mean(librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr
    ))

    rolloff = np.mean(librosa.feature.spectral_rolloff(
        y=signal,
        sr=sr
    ))

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    row = {
    "file": wav.name,
    "duration": duration,
    "samplerate": sr,
    "rms": rms,
    "peak": peak,
    "dynamic_range": dynamic_range,
    "zcr": zcr,
    "spectral_centroid": centroid,
    "bandwidth": bandwidth,
    "rolloff": rolloff,
    "silent": rms < 0.005,
    "clipped": peak > 0.99,
    }

    # Compute statistics for each MFCC coefficient
    for i in range(13):
        row[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        row[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    rows.append(row)
# ============================================================
# Create dataframe
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_DIR / "signal_statistics.csv",
    index=False
)

print(df.describe())

# ============================================================
# Histograms
# ============================================================

columns = [
    "duration",
    "rms",
    "dynamic_range",
    "zcr",
    "spectral_centroid",
    "bandwidth",
    "rolloff"
]

for col in columns:

    plt.figure(figsize=(7,4))

    plt.hist(df[col], bins=40)

    plt.title(col)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{col}.png")

    plt.close()

# ============================================================
# Random waveform plots
# ============================================================

# samples = random.sample(wav_files, min(12, len(wav_files)))

samples = random.sample(wav_files, 3)

for wav in samples:

    signal, sr = librosa.load(wav, sr=None)

    plt.figure(figsize=(12,3))

    plt.plot(signal)

    plt.title(wav.name)

    plt.xlabel("Samples")

    plt.ylabel("Amplitude")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{wav.stem}_waveform.png"
    )

    plt.close()

# ============================================================
# Random Mel Spectrograms
# ============================================================

for wav in samples:

    signal, sr = librosa.load(wav, sr=None)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)

    plt.figure(figsize=(10,4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar()

    plt.title(wav.name)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{wav.stem}_mel.png"
    )

    plt.close()

# ============================================================
# FFT plots
# ============================================================

for wav in samples:

    signal, sr = librosa.load(wav, sr=None)

    fft = np.abs(rfft(signal))

    freq = rfftfreq(len(signal), 1/sr)

    plt.figure(figsize=(10,4))

    plt.plot(freq, fft)

    plt.xlim(0,1000)

    plt.title(wav.name)

    plt.xlabel("Frequency (Hz)")

    plt.ylabel("Magnitude")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{wav.stem}_fft.png"
    )

    plt.close()

# ============================================================
# Potential problems
# ============================================================

print("\nPotential silent recordings")

print(df[df.silent][["file","rms"]])

print("\nPotential clipped recordings")

print(df[df.clipped][["file","peak"]])

# ============================================================
# Correlation matrix
# ============================================================

numeric = df.select_dtypes(include=np.number)

corr = numeric.corr()

plt.figure(figsize=(10,8))

plt.imshow(corr)

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "correlation_matrix.png"
)

plt.close()

print("\nDone!")
print(f"Results saved to {OUTPUT_DIR}")