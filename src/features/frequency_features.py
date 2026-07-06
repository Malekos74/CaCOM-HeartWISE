import numpy as np
import librosa


def extract_frequency_features(signal, sr):

    features = {}

    features["spectral_centroid"] = np.mean(
        librosa.feature.spectral_centroid(y=signal, sr=sr)
    )

    features["spectral_bandwidth"] = np.mean(
        librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    )

    features["spectral_rolloff"] = np.mean(
        librosa.feature.spectral_rolloff(y=signal, sr=sr)
    )

    features["spectral_flatness"] = np.mean(
        librosa.feature.spectral_flatness(y=signal)
    )

    # Energy in heart-relevant bands
    S = np.abs(librosa.stft(signal))**2
    freqs = librosa.fft_frequencies(sr=sr)

    bands = [(20, 50), (50, 100), (100, 200), (200, 400), (400, 800)]

    for low, high in bands:
        idx = np.where((freqs >= low) & (freqs < high))[0]
        features[f"band_energy_{low}_{high}"] = np.mean(S[idx]) if len(idx) > 0 else 0

    return features