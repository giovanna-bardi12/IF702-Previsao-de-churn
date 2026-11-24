# IF702-Previsao-de-churn

## MITRA (Amazon / AutoGluon)

Run from the repository root, with a GPU recommended:

```bash
pip install -r requirements.txt
python -m mitra.prepare_data
python -m mitra.mitra_train_baseline
python -m mitra.mitra_train --n-trials 20
python -m mitra.mitra_test
```

The default MITRA protocol balances the training and validation sets by repeating
minority-class examples; the test set retains its natural class proportions.
The original stratified splits are generated once, before balancing. For fine
tuning, MITRA first reserves 15% of the original training rows for internal
early stopping and only then oversamples the remaining training rows, preventing
copies of the same row from crossing that boundary. This intentionally differs
from randomly splitting an already oversampled training set. Training-set metrics
are reported on the original, unbalanced training split. For zero-shot, all
training rows are used as support and the internal AutoGluon score is in-sample;
external validation is used for selection.

Run the natural-prior comparison using the same prepared data and a separate
study and checkpoints:

```bash
MITRA_DATA_MODE=natural python -m mitra.mitra_train_baseline
MITRA_DATA_MODE=natural python -m mitra.mitra_train --n-trials 20
MITRA_DATA_MODE=natural python -m mitra.mitra_test
```

Each mode uses the same original splits and test rows. These are reproducible
experiment commands, not evidence that either mode performs better: run the
experiments to compare validation metrics before using the held-out test.
Changing class priors can affect probability calibration and threshold metrics.

Two preprocessing variants are prepared: `missing_encode_normalize` (default)
and `missing`, which leaves categorical encoding to AutoGluon. To run the
preprocessing comparison, set `MITRA_PREPROCESSING=missing` consistently for
training and testing; it may be combined with either `MITRA_DATA_MODE` value.
The modes and preprocessing variants use separate studies, model paths and test
results. For example:

```bash
MITRA_PREPROCESSING=missing python -m mitra.mitra_train --n-trials 20
MITRA_PREPROCESSING=missing python -m mitra.mitra_test
```

Fine-tuning uses a maximum of 10,000 epochs and patience 10. The v8 Optuna study
searches the learning rate, number of estimators, warmup steps (5 or 10),
and feature and class shuffling. It no longer wastes fine-tuning trials on
repeated zero-shot configurations. To run one separate zero-shot comparison:

```bash
python -m mitra.mitra_train_baseline --zero-shot
python -m mitra.mitra_test --zero-shot
```

`--n-trials` is the target number of completed trials, not additional trials;
increase it to extend the same study. Each
trial can take hours, especially with two estimators. OOM retries can cause MITRA
to use fewer support samples; a larger GPU may reduce retries. Set `HF_TOKEN`
through the environment or Kaggle Secrets for authenticated Hugging Face
downloads; the public checkpoint can also be downloaded without a token.

For two GPUs on the same machine, use Optuna's journal storage, with one
process per GPU. For a **new** study, the following runs 10 trial attempts per
worker (20 attempts total); repeated invocations add more attempts:

```bash
export MITRA_OPTUNA_STORAGE=journal
export MITRA_PREPROCESSING=missing  # optional; compare quality with the default
CUDA_VISIBLE_DEVICES=0 python -m mitra.mitra_train --worker-trials 10 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 python -m mitra.mitra_train --worker-trials 10 &
pid1=$!
wait "$pid0"; status0=$?
wait "$pid1"; status1=$?
test "$status0" -eq 0 && test "$status1" -eq 0
```

Use the same `MITRA_OPTUNA_STORAGE` and `MITRA_PREPROCESSING` values when
running `python -m mitra.mitra_test`. Journal and SQLite studies are independent;
`--n-trials` remains a total-completed target for a single process, whereas
`--worker-trials` is an **additional attempt count per process** and does not
enforce a global target. Failed attempts are not completed trials. The journal
file must be on a filesystem shared by the workers, and the GPU IDs must be
distinct. Parallel runs can have different Optuna suggestions and results from
sequential runs; measure runtime and validation quality instead of assuming a
speedup. Workers also share CPU, RAM, disk and W&B upload bandwidth.

To limit the search to one estimator, set `MITRA_ESTIMATORS=1` consistently
for training and testing. This uses a separate study; the default continues to
compare one and two estimators. `missing` and the one-estimator setting can
reduce resource use, but their effect on quality must be evaluated.

The held-out test is used only after selection. To test the fixed baseline, run
`python -m mitra.mitra_test --baseline`. The Optuna test command selects the
highest validation KS. Both report metrics at threshold 0.5 and at a threshold
chosen on validation to maximize TPR minus FPR. Test results include confusion
matrices, ROC and KS CDF plots in `mitra/test_results/<mode>-<preprocessing>/`.
The KS plot marks the maximum absolute CDF separation; its test-set location is
for visualization and is not used to select the classification threshold.

Training logs MSE and binary cross-entropy on the full original training split
and validation metrics to W&B. MITRA may use a training row's own label as
in-context support when predicting that row, so training scores can be optimistic.
Validation and test are the meaningful out-of-sample estimates. MITRA's binary
fine-tuning uses cross-entropy; measuring MSE does not constitute an MSE-loss
training experiment. The MSE-vs-cross-entropy training-loss comparison remains
outside this MITRA workflow.

Predictors are saved under `mitra/predictors/`, independently of W&B run folders,
and uploaded as W&B model artifacts. Optuna keeps its study in
`mitra/churn_mitra_optuna.db` by default (or a separate `.journal` file in
journal mode); baseline configuration is in `mitra/baselines/`.
After each finished trial, a consistent storage snapshot is saved in
`mitra/study_backups/` and uploaded as a separate W&B `optuna-study` artifact.
If an upload fails, the search continues and warns about the local snapshot.
Preserve the database/configuration and predictor directories when moving to a
new session, or download the relevant W&B artifact and restore the referenced
path. Older studies and baseline checkpoints remain separate from v8.
