import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "data/raw/training_data.csv"
OUTPUT_CSV = "patient_splits.csv"

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42


def main():
    df = pd.read_csv(INPUT_CSV)

    # -------------------------------------------------------
    # Step 1: Build patient-level table
    # -------------------------------------------------------
    patient_df = df.groupby("Patient ID").first().reset_index()

    patient_ids = patient_df["Patient ID"]
    labels = patient_df["Outcome"]

    # -------------------------------------------------------
    # Step 2: Train / temp split
    # -------------------------------------------------------
    train_ids, temp_ids = train_test_split(
        patient_ids,
        test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # align labels for temp split
    temp_labels = patient_df[patient_df["Patient ID"].isin(temp_ids)]["Outcome"]

    # -------------------------------------------------------
    # Step 3: Val / test split
    # -------------------------------------------------------
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE,
        stratify=temp_labels
    )

    # -------------------------------------------------------
    # Step 4: Build split dataframe
    # -------------------------------------------------------
    split_df = pd.DataFrame({
        "Patient ID": list(patient_ids),
        "split": [
            "train" if pid in set(train_ids)
            else "val" if pid in set(val_ids)
            else "test"
            for pid in patient_ids
        ]
    })

    # -------------------------------------------------------
    # Step 5: Save
    # -------------------------------------------------------
    split_df.to_csv(OUTPUT_CSV, index=False)

    print("Saved split file:", OUTPUT_CSV)
    print(split_df["split"].value_counts())
    



if __name__ == "__main__":
    main()