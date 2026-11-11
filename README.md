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
and baseline have separate names from older experiments. Training uses verbose
logging and no fixed time limit; early stopping can still result in fewer steps
than requested. The v3 Optuna study is separate from the earlier time-limited
v2 study, so trials with different training budgets are not mixed.

On GPUs with limited memory, MITRA may retry with a smaller support set after
an out-of-memory error; a GPU with more available memory can reduce retries.
AutoGluon manages this fallback internally. To authenticate Hugging Face model
downloads, set `HF_TOKEN` in the environment (for example, using Kaggle Secrets).
Authentication is optional for the public model but can improve download limits.

Use the held-out test only after selecting a model. Baseline training does not
create an Optuna study: to test that run, use
`python -m mitra.mitra_test --baseline`. The default test command requires a
completed `python -m mitra.mitra_train` run. In a new Kaggle session, restore
the Optuna SQLite database and the referenced predictor files, or rerun the
training. Raw model scores
are not guaranteed to be calibrated churn probabilities; assess and, if needed,
calibrate them on separate data before using them as probabilities.

Training logs validation metrics and confusion matrices to W&B. Predictors are
saved as artifacts and Optuna trials are tracked in the local SQLite study.
Test results are written to `mitra/test_results_natural_v2/` and logged to W&B.
