# IF702-Previsao-de-churn

## MITRA (Amazon / AutoGluon)

MITRA is integrated through AutoGluon's official tabular implementation. It
uses MITRA's pretrained weights and supports zero-shot inference and optional
fine-tuning; the Optuna search compares zero-shot runs with several fine-tuning
step counts. The first run may need internet access to download the pretrained
checkpoint from Hugging Face. A GPU is strongly recommended for fine-tuning.

```bash
pip install -r requirements.txt
python -m mitra.prepare_data
python -m mitra.mitra_train_baseline
python -m mitra.mitra_train
python -m mitra.mitra_test
```

MITRA uses its own prepared splits, preserving the original class proportions;
the splits for other models remain unchanged. Training reserves 15% of the
training split for early stopping. The separate validation split selects among
five configurations (no fine-tuning, or 10, 25, 50, or 100 steps). The saved
predictor retains its training context for in-context inference. The new study
and baseline have separate names from older experiments. The time limit and
early stopping can result in fewer steps than requested; inspect the logs.

Use the held-out test only after selecting a model. To evaluate the fixed
baseline instead, use `python -m mitra.mitra_test --baseline`. Raw model scores
are not guaranteed to be calibrated churn probabilities; assess and, if needed,
calibrate them on separate data before using them as probabilities.

Training logs validation metrics and confusion matrices to W&B. Predictors are
saved as artifacts and Optuna trials are tracked in the local SQLite study.
Test results are written to `mitra/test_results_natural_v2/` and logged to W&B.
