import numpy as np
import librosa


def extract_mfcc_features(signal, sr, n_mfcc=13):

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = {}

    for i in range(n_mfcc):
        # MFCC
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

        # Delta
        features[f"mfcc_d1_{i+1}_mean"] = np.mean(delta[i])
        features[f"mfcc_d1_{i+1}_std"] = np.std(delta[i])

        # Delta-delta
        features[f"mfcc_d2_{i+1}_mean"] = np.mean(delta2[i])
        features[f"mfcc_d2_{i+1}_std"] = np.std(delta2[i])

    return features