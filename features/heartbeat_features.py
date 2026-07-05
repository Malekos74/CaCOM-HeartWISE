import numpy as np


def extract_heartbeat_features(tsv_path):
    """
    Expected TSV format:
    time, label (S1, S2, systole, diastole)
    """

    segments = []

    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            segments.append(parts)

    times = []
    labels = []

    for row in segments:
        try:
            times.append(float(row[0]))
            labels.append(row[1])
        except:
            continue

    features = {}

    s1_times = [t for t, l in zip(times, labels) if "S1" in l]
    s2_times = [t for t, l in zip(times, labels) if "S2" in l]

    if len(s1_times) > 1:
        rr_intervals = np.diff(s1_times)
        features["heart_rate_est"] = 60 / np.mean(rr_intervals)
        features["rr_std"] = np.std(rr_intervals)
    else:
        features["heart_rate_est"] = np.nan
        features["rr_std"] = np.nan

    features["num_s1"] = len(s1_times)
    features["num_s2"] = len(s2_times)

    return features