from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputsGRF"

# Created by the main classification pipeline
PREDICTIONS_FILE = OUTPUT_DIR / "final_test_predictions.csv"
THRESHOLD_SEARCH_FILE = OUTPUT_DIR / "threshold_search.csv"

# New files created by this script
INTERPRETABLE_REPORT_FILE = OUTPUT_DIR / "interpretable_results.txt"
PATIENT_RESULTS_FILE = OUTPUT_DIR / "patient_results_interpretable.csv"


POSITIVE_LABEL = "Abnormal"
NEGATIVE_LABEL = "Normal"


def load_best_threshold() -> float:
    """
    Read the chosen threshold from threshold_search.csv.

    The previous pipeline sorts threshold_search.csv so the first row is
    the selected threshold.
    """
    if not THRESHOLD_SEARCH_FILE.exists():
        raise FileNotFoundError(
            f"Could not find threshold file:\n{THRESHOLD_SEARCH_FILE}"
        )

    threshold_table = pd.read_csv(THRESHOLD_SEARCH_FILE)

    if "threshold" not in threshold_table.columns:
        raise ValueError(
            "threshold_search.csv must contain a column named 'threshold'."
        )

    return float(threshold_table.iloc[0]["threshold"])


def load_predictions() -> pd.DataFrame:
    """
    Load patient-level test predictions.

    Required columns:
        patient_id
        true_label
        abnormal_probability

    If predicted_label is missing, it is recreated using the saved threshold.
    """
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find predictions file:\n{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(
        PREDICTIONS_FILE,
        dtype={"patient_id": str},
    )

    required = {
        "patient_id",
        "true_label",
        "abnormal_probability",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in predictions file: {sorted(missing)}"
        )

    return df


def add_interpretable_columns(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Add clear patient-level interpretation columns.
    """
    result = df.copy()

    result["predicted_label"] = result["abnormal_probability"].apply(
        lambda probability:
        POSITIVE_LABEL if probability >= threshold else NEGATIVE_LABEL
    )

    def category(row):
        actual = row["true_label"]
        predicted = row["predicted_label"]

        if actual == POSITIVE_LABEL and predicted == POSITIVE_LABEL:
            return "Correctly detected abnormal"

        if actual == POSITIVE_LABEL and predicted == NEGATIVE_LABEL:
            return "Missed abnormal"

        if actual == NEGATIVE_LABEL and predicted == POSITIVE_LABEL:
            return "False alarm"

        return "Correctly detected normal"

    result["result_category"] = result.apply(category, axis=1)

    result["prediction_correct"] = (
        result["true_label"] == result["predicted_label"]
    )

    result["abnormal_probability_percent"] = (
        result["abnormal_probability"] * 100
    ).round(1)

    return result


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate metrics where Abnormal is the positive class.
    """
    y_true = (df["true_label"] == POSITIVE_LABEL).astype(int)
    y_pred = (df["predicted_label"] == POSITIVE_LABEL).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
    }


def build_plain_language_report(
    df: pd.DataFrame,
    threshold: float,
    metrics: dict,
) -> str:
    """
    Create a readable report explaining actual and detected patients.
    """
    actual_abnormal = int(
        (df["true_label"] == POSITIVE_LABEL).sum()
    )
    actual_normal = int(
        (df["true_label"] == NEGATIVE_LABEL).sum()
    )

    predicted_abnormal = int(
        (df["predicted_label"] == POSITIVE_LABEL).sum()
    )
    predicted_normal = int(
        (df["predicted_label"] == NEGATIVE_LABEL).sum()
    )

    tp = metrics["true_positive"]
    fp = metrics["false_positive"]
    fn = metrics["false_negative"]
    tn = metrics["true_negative"]

    total = len(df)

    report = f"""
HEART-SOUND CLASSIFICATION RESULTS
==================================

Chosen abnormal-probability threshold
-------------------------------------
Threshold: {threshold:.2f}

A patient is classified as Abnormal when:

    predicted probability of Abnormal >= {threshold:.2f}

A patient is classified as Normal when:

    predicted probability of Abnormal < {threshold:.2f}


PATIENT COUNTS
==============

Total evaluated patients: {total}

Actual condition
----------------
Actually Abnormal: {actual_abnormal}
Actually Normal:   {actual_normal}

Model predictions
-----------------
Predicted Abnormal: {predicted_abnormal}
Predicted Normal:   {predicted_normal}


DETECTED VERSUS ACTUAL
======================

Abnormal patients
-----------------
Actually Abnormal:              {actual_abnormal}
Correctly detected as Abnormal: {tp}
Missed and predicted Normal:    {fn}

The model detected {tp} of the {actual_abnormal} truly abnormal patients.

This corresponds to a recall of:

    {metrics["recall"] * 100:.1f}%

Normal patients
---------------
Actually Normal:                {actual_normal}
Correctly detected as Normal:   {tn}
Incorrectly flagged Abnormal:   {fp}

The model correctly recognized {tn} of the {actual_normal} truly normal patients.

The number of false alarms was:

    {fp}


INTERPRETING ABNORMAL PREDICTIONS
================================

Patients predicted Abnormal:          {predicted_abnormal}
Predictions that were truly Abnormal: {tp}
Predictions that were false alarms:   {fp}

When the model predicted Abnormal, it was correct:

    {metrics["precision"] * 100:.1f}% of the time

This is the precision.


MAIN PERFORMANCE METRICS
========================

Precision:          {metrics["precision"] * 100:.1f}%
Recall:             {metrics["recall"] * 100:.1f}%
F1-score:           {metrics["f1"] * 100:.1f}%
Accuracy:           {metrics["accuracy"] * 100:.1f}%
Balanced accuracy:  {metrics["balanced_accuracy"] * 100:.1f}%


CONFUSION MATRIX IN PLAIN LANGUAGE
==================================

Correctly detected abnormal patients: {tp}
Missed abnormal patients:             {fn}
False alarms:                         {fp}
Correctly detected normal patients:   {tn}


HOW TO READ THESE RESULTS
=========================

Recall answers:

    Of all truly abnormal patients, how many did the model detect?

Precision answers:

    Of all patients predicted as abnormal, how many were actually abnormal?

A high recall with lower precision means that the model detects many abnormal
patients, but it also creates many false alarms.

A higher threshold will usually reduce false alarms and increase precision,
but it may also miss more abnormal patients and therefore reduce recall.
""".strip()

    return report


def print_compact_summary(
    threshold: float,
    metrics: dict,
) -> None:
    """
    Print a compact terminal summary.
    """
    print("\n" + "=" * 70)
    print("INTERPRETABLE TEST RESULTS")
    print("=" * 70)

    print(f"Chosen threshold: {threshold:.2f}\n")

    print("Actual Abnormal patients:")
    print(
        f"  Correctly detected: {metrics['true_positive']}"
    )
    print(
        f"  Missed:             {metrics['false_negative']}"
    )

    print("\nActual Normal patients:")
    print(
        f"  Correctly detected: {metrics['true_negative']}"
    )
    print(
        f"  False alarms:       {metrics['false_positive']}"
    )

    print("\nMetrics:")
    print(
        f"  Precision:         {metrics['precision']:.3f}"
    )
    print(
        f"  Recall:            {metrics['recall']:.3f}"
    )
    print(
        f"  F1-score:          {metrics['f1']:.3f}"
    )
    print(
        f"  Balanced accuracy: {metrics['balanced_accuracy']:.3f}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    threshold = load_best_threshold()
    predictions = load_predictions()

    interpreted = add_interpretable_columns(
        predictions,
        threshold,
    )

    metrics = calculate_metrics(interpreted)

    report = build_plain_language_report(
        interpreted,
        threshold,
        metrics,
    )

    # Save patient-level table, ordered so errors are easy to inspect.
    category_order = {
        "Missed abnormal": 0,
        "False alarm": 1,
        "Correctly detected abnormal": 2,
        "Correctly detected normal": 3,
    }

    interpreted["_category_order"] = interpreted[
        "result_category"
    ].map(category_order)

    interpreted = interpreted.sort_values(
        by=[
            "_category_order",
            "abnormal_probability",
        ],
        ascending=[True, False],
    ).drop(columns="_category_order")

    columns = [
        "patient_id",
        "true_label",
        "predicted_label",
        "abnormal_probability_percent",
        "result_category",
        "prediction_correct",
    ]

    interpreted[columns].to_csv(
        PATIENT_RESULTS_FILE,
        index=False,
    )

    INTERPRETABLE_REPORT_FILE.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print_compact_summary(threshold, metrics)

    print("\nSaved readable report:")
    print(INTERPRETABLE_REPORT_FILE)

    print("\nSaved patient-level results:")
    print(PATIENT_RESULTS_FILE)


if __name__ == "__main__":
    main()