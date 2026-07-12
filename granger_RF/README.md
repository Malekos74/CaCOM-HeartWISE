# Full Heart-Sound Pipeline: Raw WAV → Acoustic + Granger → Tuned Random Forest

This version does **not** require precomputed feature CSV files.

It starts from the raw CirCor `.wav` and `.txt` files, recomputes the features, builds patient-level acoustic and Granger features, and trains a tuned Random Forest.

## Full workflow

```text
Raw CirCor WAV files
        ↓
Frame-level acoustic time series
        ↓
Acoustic patient features
        +
Granger patient features
        ↓
Merge using patient_id
        ↓
Use existing patient_splits.csv
        ↓
Feature selection
        ↓
Parallel Random Forest tuning
        ↓
Validation threshold tuning
        ↓
Final test evaluation
```

## Existing split file

The pipeline uses the split that you already created:

```text
/Users/marammarzouki/CaCOM-HeartWISE/data/processed/patient_splits.csv
```

It does not create a new split.

Expected columns:

```text
patient_id,split
```

## Raw data path

The default raw data path is:

```text
/Users/marammarzouki/CaCOM-HeartWISE/data/training_data
```

Change `DATA_DIR` in `config.py` if your `.wav` and `.txt` files are elsewhere.

## Runtime

The pipeline is optimized for a 16 GB laptop:

- recording-level extraction is parallelized;
- Granger computation is parallelized by recording;
- Random Forest search is parallelized;
- nested parallelism is avoided;
- results are cached;
- existing outputs are reused.

The default `FAST_MODE=True` uses 9 acoustic time series and `MAX_LAG=2`. This is much faster than testing all 17 time series and 272 directed pairs.

A runtime below 30 minutes cannot be guaranteed because it depends on the number and length of recordings and the MacBook processor. Full 17-feature Granger recomputation may take considerably longer.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Recompute everything

By default, existing outputs are reused.

To force all stages to recompute, set this in `config.py`:

```python
FORCE_RECOMPUTE = True
```

## Outputs

```text
outputs/frame_features/
outputs/granger_edges/
outputs/labels.csv
outputs/acoustic_patient_features.csv
outputs/granger_patient_features.csv
outputs/best_random_forest.joblib
outputs/selected_features.csv
outputs/threshold_search.csv
outputs/final_test_predictions.csv
outputs/model_summary.txt
```
