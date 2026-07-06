import pandas as pd
from tqdm import tqdm

from features.loader import load_audio, get_all_files
from features.extractor import extract_features

DATA_DIR = "data/training_data"

# ---------------------------------------------------------
# Load metadata (patient-level table)
# ---------------------------------------------------------
metadata_df = pd.read_csv("data/training_data.csv")
metadata_df["Patient ID"] = metadata_df["Patient ID"].astype(str)

metadata_lookup = metadata_df.set_index("Patient ID").to_dict(orient="index")

# ---------------------------------------------------------
# Load patient splits
# ---------------------------------------------------------
splits_df = pd.read_csv("patient_splits.csv")
splits_df["Patient ID"] = splits_df["Patient ID"].astype(str)

split_lookup = dict(zip(splits_df["Patient ID"], splits_df["split"]))

# ---------------------------------------------------------
# Get all files
# ---------------------------------------------------------
files = get_all_files(DATA_DIR)

rows = []

# ---------------------------------------------------------
# Feature extraction with progress bar
# ---------------------------------------------------------
for file in tqdm(files, desc="Extracting features"):

    signal, sr = load_audio(file)
    feature_dict = extract_features(signal, sr)

    # Parse filename: 2530_AV.wav
    patient_id = file.stem.split("_")[0]
    recording_location = file.stem.split("_")[1]

    # Safety checks (avoid crashes)
    if patient_id not in metadata_lookup:
        print(f"Warning: patient {patient_id} not in metadata, skipping {file.name}")
        continue

    if patient_id not in split_lookup:
        print(f"Warning: patient {patient_id} not in split file, skipping {file.name}")
        continue

    metadata = metadata_lookup[patient_id]

    # -----------------------------------------------------
    # Build ordered row
    # -----------------------------------------------------
    row = {
        # 1. ID first
        "Patient ID": patient_id,

        # 2. Label second
        "Outcome": metadata["Outcome"],

        # 3. Split (useful for debugging/training)
        "split": split_lookup[patient_id],

        # 4. Metadata
        "Age": metadata["Age"],
        "Sex": metadata["Sex"],
        "Height": metadata["Height"],
        "Weight": metadata["Weight"],
        "Pregnancy status": metadata["Pregnancy status"],
        "Murmur": metadata["Murmur"],

        # 5. Recording info
        "recording_location": recording_location,
        "file": file.name,

        # Optional IDs / tracking
        "Campaign": metadata["Campaign"],
        "Additional ID": metadata["Additional ID"],
    }

    # 6. Add extracted audio features
    row.update(feature_dict)

    rows.append(row)

# ---------------------------------------------------------
# Build dataframe
# ---------------------------------------------------------
df = pd.DataFrame(rows)

# Ensure column order is preserved exactly
base_cols = [
    "Patient ID",
    "Outcome",
    "split",
    "Age",
    "Sex",
    "Height",
    "Weight",
    "Pregnancy status",
    "Murmur",
    "recording_location",
    "file",
    "Campaign",
    "Additional ID",
]

feature_cols = [c for c in df.columns if c not in base_cols]

df = df[base_cols + feature_cols]

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
df.to_csv("feature_dataset.csv", index=False)

print("\nDone!")
print("Shape:", df.shape)
print(df["split"].value_counts())