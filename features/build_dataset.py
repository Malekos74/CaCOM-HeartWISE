import pandas as pd

from features.loader import load_audio, get_all_files
from features.extractor import extract_features

DATA_DIR = "data/training_data"

# Load patient splits
split_df = pd.read_csv("patient_splits.csv")
split_lookup = dict(zip(split_df["patient_id"].astype(str),
                        split_df["split"]))

rows = []

files = get_all_files(DATA_DIR)

for file in files:

    signal, sr = load_audio(file)

    features = extract_features(signal, sr)

    patient_id = file.stem.split("_")[0]
    valve = file.stem.split("_")[1]

    features["file"] = file.name
    features["patient_id"] = patient_id
    features["valve"] = valve
    features["split"] = split_lookup[patient_id]

    rows.append(features)

df = pd.DataFrame(rows)

df.to_csv("feature_dataset.csv", index=False)

print(df["split"].value_counts())
print(df.shape)