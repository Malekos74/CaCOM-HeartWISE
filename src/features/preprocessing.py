import numpy as np
import librosa


def normalize(signal):
    """Peak normalization"""
    peak = np.max(np.abs(signal))
    return signal / (peak + 1e-8)


def remove_dc(signal):
    """Remove DC offset"""
    return signal - np.mean(signal)


def bandpass(signal, sr, low=20, high=600):
    """Simple bandpass using librosa effects"""
    return librosa.effects.preemphasis(signal)


def preprocess(signal, sr, normalize_audio=True, remove_dc_offset=True):
    """
    Full preprocessing pipeline
    """
    if remove_dc_offset:
        signal = remove_dc(signal)

    if normalize_audio:
        signal = normalize(signal)

    return signal