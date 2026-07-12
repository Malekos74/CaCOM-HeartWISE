from __future__ import annotations

import time
from pathlib import Path

import joblib
import librosa
import networkx as nx
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from statsmodels.tsa.stattools import grangercausalitytests

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import *


POSITIVE_LABEL = "Abnormal"
NEGATIVE_LABEL = "Normal"
EDGE_COLUMNS = ["source", "target", "lag", "p_value", "f_stat"]


def read_outcome(txt_path: Path) -> str | None:
    with txt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#Outcome:"):
                return line.split(":", 1)[1].strip()
    return None


def patient_id_from_recording(name: str) -> str:
    return str(name).split("_")[0]


def normalize_split(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"val", "valid", "validation"}:
        return "validation"
    return value


def spectral_entropy(y: np.ndarray) -> np.ndarray:
    spectrum = np.abs(
        librosa.stft(
            y,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
    ) ** 2

    probabilities = spectrum / (
        spectrum.sum(axis=0, keepdims=True) + 1e-12
    )

    return -np.sum(
        probabilities * np.log(probabilities + 1e-12),
        axis=0,
    )


def band_energy(
    y: np.ndarray,
    sample_rate: int,
    low: float,
    high: float,
) -> np.ndarray:
    spectrum = np.abs(
        librosa.stft(
            y,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
    ) ** 2

    frequencies = librosa.fft_frequencies(
        sr=sample_rate,
        n_fft=FRAME_LENGTH,
    )

    mask = (frequencies >= low) & (frequencies < high)
    return spectrum[mask].sum(axis=0)


def extract_one_recording(wav_path: Path) -> tuple[str, str | None]:
    output_path = FRAME_FEATURE_DIR / f"{wav_path.stem}.csv"

    if output_path.exists() and not FORCE_RECOMPUTE:
        return wav_path.stem, None

    try:
        y, sample_rate = librosa.load(
            wav_path,
            sr=None,
            mono=True,
        )

        features = {
            "rms": librosa.feature.rms(
                y=y,
                frame_length=FRAME_LENGTH,
                hop_length=HOP_LENGTH,
            )[0],
            "centroid": librosa.feature.spectral_centroid(
                y=y,
                sr=sample_rate,
                n_fft=FRAME_LENGTH,
                hop_length=HOP_LENGTH,
            )[0],
            "bandwidth": librosa.feature.spectral_bandwidth(
                y=y,
                sr=sample_rate,
                n_fft=FRAME_LENGTH,
                hop_length=HOP_LENGTH,
            )[0],
            "zcr": librosa.feature.zero_crossing_rate(
                y,
                frame_length=FRAME_LENGTH,
                hop_length=HOP_LENGTH,
            )[0],
            "entropy": spectral_entropy(y),
            "energy_20_100": band_energy(y, sample_rate, 20, 100),
            "energy_100_250": band_energy(y, sample_rate, 100, 250),
            "energy_250_500": band_energy(y, sample_rate, 250, 500),
            "energy_500_1000": band_energy(y, sample_rate, 500, 1000),
        }

        if not FAST_MODE:
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sample_rate,
                n_mfcc=N_MFCC,
                n_fft=FRAME_LENGTH,
                hop_length=HOP_LENGTH,
            )

            for index in range(N_MFCC):
                features[f"mfcc_{index + 1}"] = mfcc[index]

        common_length = min(len(values) for values in features.values())

        frame_df = pd.DataFrame(
            {
                name: values[:common_length]
                for name, values in features.items()
            }
        )

        frame_df.to_csv(output_path, index=False)
        return wav_path.stem, None

    except Exception as error:
        return wav_path.stem, str(error)


def stage_extract_frame_features() -> None:
    print("\nSTAGE 1 — FRAME-LEVEL ACOUSTIC FEATURES")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(DATA_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No patient .txt files were found in {DATA_DIR}"
        )

    labels = []
    wav_files = []

    for txt_path in txt_files:
        patient_id = txt_path.stem
        outcome = read_outcome(txt_path)

        if outcome not in {NORMAL := "Normal", POSITIVE_LABEL}:
            continue

        labels.append(
            {
                "patient_id": patient_id,
                "label": outcome,
            }
        )

        wav_files.extend(
            sorted(DATA_DIR.glob(f"{patient_id}_*.wav"))
        )

    pd.DataFrame(labels).to_csv(LABELS_FILE, index=False)

    print("Patients with labels:", len(labels))
    print("WAV recordings:", len(wav_files))
    print("Parallel workers:", PREPROCESS_N_JOBS)
    print("Feature time series:", len(FEATURE_NAMES))

    results = Parallel(
        n_jobs=PREPROCESS_N_JOBS,
        backend="loky",
        verbose=10,
    )(
        delayed(extract_one_recording)(wav_path)
        for wav_path in wav_files
    )

    failures = [
        (recording, error)
        for recording, error in results
        if error is not None
    ]

    print("Frame feature files:", len(list(FRAME_FEATURE_DIR.glob("*.csv"))))
    print("Failures:", len(failures))

    if failures:
        pd.DataFrame(
            failures,
            columns=["recording", "error"],
        ).to_csv(
            OUTPUT_DIR / "feature_extraction_failures.csv",
            index=False,
        )


def test_granger_pair(
    frame_df: pd.DataFrame,
    cause: str,
    effect: str,
) -> tuple[int, float, float] | None:
    data = (
        frame_df[[effect, cause]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(data) < max(30, MAX_LAG + 5):
        return None

    if data[cause].nunique() < 3 or data[effect].nunique() < 3:
        return None

    try:
        results = grangercausalitytests(
            data,
            maxlag=MAX_LAG,
            verbose=False,
        )

        candidates = []

        for lag, result in results.items():
            f_statistic = result[0]["ssr_ftest"][0]
            p_value = result[0]["ssr_ftest"][1]

            candidates.append(
                (lag, p_value, f_statistic)
            )

        best_lag, best_p, best_f = min(
            candidates,
            key=lambda item: item[1],
        )

        if np.isfinite(best_p) and best_p < P_VALUE_THRESHOLD:
            return best_lag, best_p, best_f

    except Exception:
        return None

    return None


def granger_one_recording(
    feature_file: Path,
) -> tuple[str, int, str | None]:
    output_path = GRANGER_DIR / feature_file.name

    if output_path.exists() and not FORCE_RECOMPUTE:
        try:
            existing = pd.read_csv(output_path)
            return feature_file.stem, len(existing), None
        except Exception:
            pass

    try:
        frame_df = pd.read_csv(feature_file)
        available = [
            feature
            for feature in FEATURE_NAMES
            if feature in frame_df.columns
        ]

        edges = []

        for cause in available:
            for effect in available:
                if cause == effect:
                    continue

                result = test_granger_pair(
                    frame_df,
                    cause,
                    effect,
                )

                if result is not None:
                    lag, p_value, f_statistic = result

                    edges.append(
                        {
                            "source": cause,
                            "target": effect,
                            "lag": lag,
                            "p_value": p_value,
                            "f_stat": f_statistic,
                        }
                    )

        pd.DataFrame(
            edges,
            columns=EDGE_COLUMNS,
        ).to_csv(
            output_path,
            index=False,
        )

        return feature_file.stem, len(edges), None

    except Exception as error:
        # Preserve the recording with an empty valid edge file.
        pd.DataFrame(
            columns=EDGE_COLUMNS
        ).to_csv(
            output_path,
            index=False,
        )

        return feature_file.stem, 0, str(error)


def stage_granger() -> None:
    print("\nSTAGE 2 — GRANGER FEATURES")
    print("=" * 80)

    GRANGER_DIR.mkdir(parents=True, exist_ok=True)
    feature_files = sorted(FRAME_FEATURE_DIR.glob("*.csv"))

    if not feature_files:
        raise FileNotFoundError(
            "No frame feature files exist. Stage 1 must run first."
        )

    possible_pairs = len(FEATURE_NAMES) * (len(FEATURE_NAMES) - 1)

    print("Recordings:", len(feature_files))
    print("Time series:", len(FEATURE_NAMES))
    print("Directed pairs per recording:", possible_pairs)
    print("Maximum lag:", MAX_LAG)
    print("Parallel workers:", PREPROCESS_N_JOBS)

    results = Parallel(
        n_jobs=PREPROCESS_N_JOBS,
        backend="loky",
        verbose=10,
    )(
        delayed(granger_one_recording)(feature_file)
        for feature_file in feature_files
    )

    failures = [
        (recording, error)
        for recording, _, error in results
        if error is not None
    ]

    empty_count = sum(
        1
        for _, edge_count, _ in results
        if edge_count == 0
    )

    print("Granger files:", len(list(GRANGER_DIR.glob("*.csv"))))
    print("Zero-edge recordings:", empty_count)
    print("Failures preserved as zero-edge recordings:", len(failures))

    if failures:
        pd.DataFrame(
            failures,
            columns=["recording", "error"],
        ).to_csv(
            OUTPUT_DIR / "granger_failures.csv",
            index=False,
        )


def summarize_recording(feature_file: Path) -> dict:
    frame_df = pd.read_csv(feature_file)
    output = {}

    for column in FEATURE_NAMES:
        if column not in frame_df.columns:
            continue

        series = frame_df[column]

        output[f"{column}_mean"] = series.mean()
        output[f"{column}_std"] = series.std()
        output[f"{column}_median"] = series.median()
        output[f"{column}_q25"] = series.quantile(0.25)
        output[f"{column}_q75"] = series.quantile(0.75)
        output[f"{column}_min"] = series.min()
        output[f"{column}_max"] = series.max()

    output["patient_id"] = patient_id_from_recording(feature_file.stem)
    return output


def empty_graph_features() -> dict:
    output = {
        "n_edges": 0,
        "n_nodes": 0,
        "density": 0,
        "mean_lag": 0,
        "mean_p_value": 1,
        "mean_f_stat": 0,
        "max_f_stat": 0,
        "max_in_degree": 0,
        "max_out_degree": 0,
    }

    for source in FEATURE_NAMES:
        for target in FEATURE_NAMES:
            if source == target:
                continue

            prefix = f"{source}_to_{target}"

            output[f"{prefix}_exists"] = 0
            output[f"{prefix}_lag"] = 0
            output[f"{prefix}_pvalue"] = 1
            output[f"{prefix}_fstat"] = 0

    return output


def summarize_granger_recording(feature_file: Path) -> dict:
    edge_file = GRANGER_DIR / feature_file.name
    output = empty_graph_features()

    if edge_file.exists():
        edges = pd.read_csv(edge_file)
    else:
        edges = pd.DataFrame(columns=EDGE_COLUMNS)

    if not edges.empty:
        graph = nx.DiGraph()

        for _, row in edges.iterrows():
            source = row["source"]
            target = row["target"]
            graph.add_edge(source, target)

            prefix = f"{source}_to_{target}"
            output[f"{prefix}_exists"] = 1
            output[f"{prefix}_lag"] = row["lag"]
            output[f"{prefix}_pvalue"] = row["p_value"]
            output[f"{prefix}_fstat"] = row["f_stat"]

        output["n_edges"] = len(edges)
        output["n_nodes"] = graph.number_of_nodes()
        output["density"] = (
            nx.density(graph)
            if graph.number_of_nodes() > 1
            else 0
        )
        output["mean_lag"] = edges["lag"].mean()
        output["mean_p_value"] = edges["p_value"].mean()
        output["mean_f_stat"] = edges["f_stat"].mean()
        output["max_f_stat"] = edges["f_stat"].max()
        output["max_in_degree"] = max(
            dict(graph.in_degree()).values(),
            default=0,
        )
        output["max_out_degree"] = max(
            dict(graph.out_degree()).values(),
            default=0,
        )

    output["patient_id"] = patient_id_from_recording(feature_file.stem)
    return output


def stage_patient_features() -> None:
    print("\nSTAGE 3 — PATIENT-LEVEL FEATURES")
    print("=" * 80)

    frame_files = sorted(FRAME_FEATURE_DIR.glob("*.csv"))
    labels = pd.read_csv(
        LABELS_FILE,
        dtype={"patient_id": str},
    )

    if FORCE_RECOMPUTE or not ACOUSTIC_FEATURES_FILE.exists():
        acoustic_rows = Parallel(
            n_jobs=PREPROCESS_N_JOBS,
            backend="loky",
        )(
            delayed(summarize_recording)(feature_file)
            for feature_file in frame_files
        )

        acoustic_recordings = pd.DataFrame(acoustic_rows)
        acoustic_patients = (
            acoustic_recordings
            .groupby("patient_id", as_index=False)
            .mean(numeric_only=True)
        )

        acoustic_patients = acoustic_patients.merge(
            labels,
            on="patient_id",
            how="inner",
        )

        acoustic_patients.to_csv(
            ACOUSTIC_FEATURES_FILE,
            index=False,
        )
    else:
        acoustic_patients = pd.read_csv(
            ACOUSTIC_FEATURES_FILE,
            dtype={"patient_id": str},
        )

    if FORCE_RECOMPUTE or not GRANGER_FEATURES_FILE.exists():
        granger_rows = Parallel(
            n_jobs=PREPROCESS_N_JOBS,
            backend="loky",
        )(
            delayed(summarize_granger_recording)(feature_file)
            for feature_file in frame_files
        )

        granger_recordings = pd.DataFrame(granger_rows)
        granger_patients = (
            granger_recordings
            .groupby("patient_id", as_index=False)
            .mean(numeric_only=True)
        )

        granger_patients = granger_patients.merge(
            labels,
            on="patient_id",
            how="inner",
        )

        granger_patients.to_csv(
            GRANGER_FEATURES_FILE,
            index=False,
        )
    else:
        granger_patients = pd.read_csv(
            GRANGER_FEATURES_FILE,
            dtype={"patient_id": str},
        )

    common = set(acoustic_patients["patient_id"]) & set(
        granger_patients["patient_id"]
    )

    print("Acoustic patients:", len(acoustic_patients))
    print("Granger patients:", len(granger_patients))
    print("Common patients:", len(common))
    print("Acoustic features:", acoustic_patients.shape[1] - 2)
    print("Granger features:", granger_patients.shape[1] - 2)


def load_model_data() -> pd.DataFrame:
    acoustic = pd.read_csv(
        ACOUSTIC_FEATURES_FILE,
        dtype={"patient_id": str},
    )

    granger = pd.read_csv(
        GRANGER_FEATURES_FILE,
        dtype={"patient_id": str},
    ).drop(
        columns=["label"],
        errors="ignore",
    )

    splits = pd.read_csv(
        SPLITS_FILE,
        dtype={"patient_id": str},
    )

    splits["split"] = splits["split"].apply(normalize_split)

    combined = acoustic.merge(
        granger,
        on="patient_id",
        how="inner",
    ).merge(
        splits[["patient_id", "split"]],
        on="patient_id",
        how="inner",
    )

    return combined


def build_model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("selector", SelectKBest(score_func=f_classif, k=125)),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=FOREST_N_JOBS,
                ),
            ),
        ]
    )


def parameter_space() -> dict:
    return {
        "selector__k": [50, 75, 125, 200, 300],
        "model__n_estimators": [250, 400, 600],
        "model__max_depth": [None, 12, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4, 7],
        "model__max_features": ["sqrt", "log2", 0.25, 0.40],
        "model__bootstrap": [True, False],
        "model__criterion": ["gini", "entropy"],
        "model__class_weight": [
            "balanced",
            "balanced_subsample",
            {0: 1.0, 1: 1.15},
            {0: 1.0, 1: 1.30},
        ],
    }


def metrics_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "precision_recall_floor": min(precision, recall),
    }


def choose_threshold(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[float, pd.DataFrame]:
    probabilities = model.predict_proba(X_validation)[:, 1]

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX + THRESHOLD_STEP / 2,
        THRESHOLD_STEP,
    )

    rows = [
        metrics_at_threshold(
            y_validation.to_numpy(),
            probabilities,
            float(round(threshold, 2)),
        )
        for threshold in thresholds
    ]

    results = pd.DataFrame(rows).sort_values(
        ["f1", "precision_recall_floor", "balanced_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    return float(results.iloc[0]["threshold"]), results


def evaluate_model(
    name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y_text: pd.Series,
    threshold: float,
) -> tuple[dict, pd.DataFrame]:
    probabilities = model.predict_proba(X)[:, 1]
    predictions_binary = (probabilities >= threshold).astype(int)
    y_binary = (y_text == POSITIVE_LABEL).astype(int).to_numpy()

    metrics = metrics_at_threshold(
        y_binary,
        probabilities,
        threshold,
    )

    predictions_text = np.where(
        predictions_binary == 1,
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
    )

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)
    print(f"Threshold: {threshold:.2f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")

    print(
        classification_report(
            y_text,
            predictions_text,
            labels=[POSITIVE_LABEL, NEGATIVE_LABEL],
            zero_division=0,
        )
    )

    print(
        confusion_matrix(
            y_text,
            predictions_text,
            labels=[POSITIVE_LABEL, NEGATIVE_LABEL],
        )
    )

    predictions = pd.DataFrame(
        {
            "true_label": y_text.to_numpy(),
            "abnormal_probability": probabilities,
            "predicted_label": predictions_text,
        },
        index=X.index,
    )

    return metrics, predictions


def stage_random_forest() -> None:
    print("\nSTAGE 4 — TUNED RANDOM FOREST")
    print("=" * 80)

    data = load_model_data()

    feature_columns = [
        column
        for column in data.columns
        if column not in {"patient_id", "label", "split"}
    ]

    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()

    print("Input features:", len(feature_columns))
    print("Train:", len(train), train["label"].value_counts().to_dict())
    print(
        "Validation:",
        len(validation),
        validation["label"].value_counts().to_dict(),
    )
    print("Test:", len(test), test["label"].value_counts().to_dict())

    X_train = train[feature_columns]
    y_train = (train["label"] == POSITIVE_LABEL).astype(int)

    X_validation = validation[feature_columns]
    y_validation = (validation["label"] == POSITIVE_LABEL).astype(int)

    X_test = test[feature_columns]

    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=build_model_pipeline(),
        param_distributions=parameter_space(),
        n_iter=N_ITER_SEARCH,
        scoring="f1",
        cv=cross_validation,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=SEARCH_N_JOBS,
        pre_dispatch=PRE_DISPATCH,
        verbose=2,
        return_train_score=False,
        error_score="raise",
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("\nBest CV F1:", round(search.best_score_, 4))
    print("Best parameters:")
    for key, value in search.best_params_.items():
        print(f"  {key}: {value}")

    pd.DataFrame(search.cv_results_).to_csv(
        OUTPUT_DIR / "search_results.csv",
        index=False,
    )

    threshold, threshold_table = choose_threshold(
        best_model,
        X_validation,
        y_validation,
    )

    threshold_table.to_csv(
        OUTPUT_DIR / "threshold_search.csv",
        index=False,
    )

    validation_metrics, _ = evaluate_model(
        "Validation results",
        best_model,
        X_validation,
        validation["label"],
        threshold,
    )

    test_metrics, test_predictions = evaluate_model(
        "Final test results",
        best_model,
        X_test,
        test["label"],
        threshold,
    )

    test_predictions.insert(
        0,
        "patient_id",
        test.loc[test_predictions.index, "patient_id"],
    )

    test_predictions.to_csv(
        OUTPUT_DIR / "final_test_predictions.csv",
        index=False,
    )

    variance = best_model.named_steps["variance"]
    selector = best_model.named_steps["selector"]

    names_after_variance = np.asarray(feature_columns)[
        variance.get_support()
    ]
    selected_names = names_after_variance[
        selector.get_support()
    ]

    pd.DataFrame(
        {"feature": selected_names}
    ).to_csv(
        OUTPUT_DIR / "selected_features.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": best_model,
            "threshold": threshold,
            "feature_columns": feature_columns,
        },
        OUTPUT_DIR / "best_random_forest.joblib",
    )

    summary = [
        f"FAST_MODE: {FAST_MODE}",
        f"Time-series features: {len(FEATURE_NAMES)}",
        f"Maximum Granger lag: {MAX_LAG}",
        f"Best CV F1: {search.best_score_:.4f}",
        f"Threshold: {threshold:.2f}",
        "",
        "Validation:",
        f"Precision: {validation_metrics['precision']:.4f}",
        f"Recall: {validation_metrics['recall']:.4f}",
        f"F1: {validation_metrics['f1']:.4f}",
        f"Balanced accuracy: {validation_metrics['balanced_accuracy']:.4f}",
        "",
        "Test:",
        f"Precision: {test_metrics['precision']:.4f}",
        f"Recall: {test_metrics['recall']:.4f}",
        f"F1: {test_metrics['f1']:.4f}",
        f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}",
    ]

    (OUTPUT_DIR / "model_summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )


def main() -> None:
    started = time.perf_counter()

    print("FULL HEART-SOUND PIPELINE")
    print("=" * 80)
    print("Raw data:", DATA_DIR)
    print("Split CSV:", SPLITS_FILE)
    print("FAST_MODE:", FAST_MODE)
    print("Feature time series:", FEATURE_NAMES)

    stage_extract_frame_features()
    stage_granger()
    stage_patient_features()
    stage_random_forest()

    elapsed_minutes = (time.perf_counter() - started) / 60

    print("\n" + "=" * 80)
    print(f"TOTAL RUNTIME: {elapsed_minutes:.2f} minutes")
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
