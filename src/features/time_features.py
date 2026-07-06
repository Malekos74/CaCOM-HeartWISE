import numpy as np
import librosa
from scipy.stats import skew, kurtosis


def extract_time_features(signal):
    features = {}

    features["rms"] = np.sqrt(np.mean(signal**2))
    features["peak"] = np.max(np.abs(signal))
    features["variance"] = np.var(signal)
    features["mean"] = np.mean(signal)
    features["std"] = np.std(signal)

    features["skew"] = skew(signal)
    features["kurtosis"] = kurtosis(signal)

    features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(signal))

    # dynamic range
    features["dynamic_range"] = features["peak"] / (features["rms"] + 1e-8)

    return features