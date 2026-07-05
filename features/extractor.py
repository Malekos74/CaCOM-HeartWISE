from features.preprocessing import preprocess
from features.time_features import extract_time_features
from features.frequency_features import extract_frequency_features
from features.mfcc_features import extract_mfcc_features


def extract_features(signal, sr):

    signal = preprocess(signal, sr)

    features = {}

    features.update(extract_time_features(signal))
    features.update(extract_frequency_features(signal, sr))
    features.update(extract_mfcc_features(signal, sr))

    return features