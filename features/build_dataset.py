import pandas as pd
from features.loader import load_audio, get_all_files
from features.extractor import extract_features

DATA_DIR = "data/training_data"

rows = []

files = get_all_files(DATA_DIR)

for file in files:

    signal, sr = load_audio(file)

    features = extract_features(signal, sr)

    features["file"] = file.name

    rows.append(features)

df = pd.DataFrame(rows)

df.to_csv("feature_dataset.csv", index=False)

print("Saved dataset:", df.shape)