# IF702-Previsao-de-churn

## MITRA (Amazon / AutoGluon)

MITRA is integrated through AutoGluon's official tabular implementation. It
uses MITRA's pretrained weights and supports zero-shot inference and optional
fine-tuning; the Optuna search compares zero-shot runs with several fine-tuning
step counts. The first run may need internet access to download the pretrained
checkpoint from Hugging Face.

```bash
pip install -r requirements.txt
python -m mitra.mitra_train_baseline
python -m mitra.mitra_test --baseline
python -m mitra.mitra_train
python -m mitra.mitra_test --top-k 5
```

Training logs validation metrics and confusion matrices to W&B. Optuna trial
predictors are saved with model artifacts and tracked in the local SQLite
study under `mitra/`. Test metrics and confusion matrix images are written to
`mitra/test_results/` and logged to W&B.
