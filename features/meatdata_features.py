import pandas as pd


def extract_metadata_features(metadata_row):
    """
    metadata_row = row from patients dataframe
    """

    features = {}

    features["age"] = metadata_row.get("Age", None)
    features["sex"] = metadata_row.get("Sex", None)
    features["weight"] = metadata_row.get("Weight", None)
    features["height"] = metadata_row.get("Height", None)

    return features