# Experiment Code Guide

## Purpose
This repository contains an experiment framework for future negative electricity price event prediction.
The task is binary classification, not price regression.

Core label:

`y_t^(h) = 1 if price_(t+h) < 0 else 0`

Core causal rule:

- Inputs may only use information available up to anchor time `t`
- Samples are assigned to train, validation, and test by `target_time`, not by `anchor_time`
- Any preprocessing fitted on data must be fitted on the training portion only

## Main entrypoints

- `run_negative_price_experiments.py`
  Main CLI entrypoint for running the implemented experiment IDs
- `negative_price_experiments/config.py`
  Fixed experiment definitions, country sets, folds, and transfer protocol
- `negative_price_experiments/data.py`
  Dataset loading, feature preparation, missing-value handling, sample generation, and tabular or sequence dataset builders
- `negative_price_experiments/models.py`
  Baselines, tree-model wrappers, calibration helpers, and sequence training utilities
- `negative_price_experiments/metrics.py`
  PR-AUC, ROC-AUC, F1, threshold selection, and grouped summaries
- `negative_price_experiments/pipeline.py`
  End-to-end experiment runners for standard experiments and transfer experiments

## Experiment IDs

- `E1`: main 20-market public-feature experiment, `h=6`
- `E2`: main 20-market public-feature experiment, `h=24`
- `E3`: 15-market renewables-enhanced experiment, `h=6`
- `E4`: 15-market renewables-enhanced experiment, `h=24`
- `E5`: 7-market cross-border-flow experiment, `h=6`
- `E6`: transfer-learning experiment on public features with GRU only
- `E7`: main 20-market public-feature benchmark with `LightGBM`, `h=6`
- `E8`: main 20-market public-feature benchmark with `CatBoost`, `h=6`
- `E9`: main 20-market public-feature benchmark with weighted and calibrated `XGBoost`, `h=6`
- `E10`: main 20-market public-feature benchmark with `PatchTST`, `h=6`

Planned next-phase experiment IDs:

- `E11`: main 20-market public-feature deep-learning experiment with `GRU`, `window=120`, `h=6`
- `E12`: main 20-market public-feature deep-learning experiment with `GRU`, `window=168`, `h=6`
- `E13`: main 20-market public-feature deep-learning experiment with `TCN`, `window=168`, `h=6`
- `E14`: imbalance-aware extension of the best deep model selected from `E11-E13`
- `E15`: renewables-enhanced extension of the best deep model selected from `E11-E13`, `h=6`
- `E16`: cross-border-flow extension of the best deep model selected from `E11-E13`, `h=6`

Important:

- `E11-E16` are currently planned experiments, not yet implemented defaults
- `E14-E16` should not be fixed to one backbone in advance
- the backbone and window used by `E14-E16` should be chosen based on the results of `E11-E13`

All default experiment definitions are created in:

- `build_default_experiment_configs(...)`
- `build_default_transfer_config(...)`

## Execution

Run the standard experiment set:

```bash
python3 run_negative_price_experiments.py \
  --data-file ALL_COUNTRIES_2024_2025_WITH_ERA5.csv \
  --output-dir experiment_outputs \
  --experiments E1,E2,E3,E4
```

Run the transfer experiment:

```bash
python3 run_negative_price_experiments.py \
  --data-file ALL_COUNTRIES_2024_2025_WITH_ERA5.csv \
  --output-dir experiment_outputs \
  --experiments E6
```

Run the advanced post-`E6` model experiments:

```bash
python3 run_negative_price_experiments.py \
  --data-file ALL_COUNTRIES_2024_2025_WITH_ERA5.csv \
  --output-dir experiment_outputs \
  --experiments E7,E8,E9,E10
```

Planned next deep-learning phase:

- first run `E11-E13` to select the strongest deep sequence model and history window
- then define `E14-E16` on top of that selected backbone

If optional ML dependencies are missing and you only want available models:

```bash
python3 run_negative_price_experiments.py \
  --data-file ALL_COUNTRIES_2024_2025_WITH_ERA5.csv \
  --output-dir experiment_outputs \
  --experiments E1 \
  --skip-unavailable-models
```

Install experiment dependencies with:

```bash
python3 -m pip install -r requirements-experiments.txt
```

Device behavior:

- PyTorch sequence models prefer `cuda`, then `mps`, then `cpu`
- XGBoost uses GPU only when `cuda` is available
- Apple `mps` is not used by XGBoost

## Data and sample construction

The merged source dataset is expected to be:

- `ALL_COUNTRIES_2024_2025_WITH_ERA5.csv`

Required assumptions:

- Hourly panel
- `UTC` timestamps
- Unique `(country, time)` rows
- Weather columns already merged in

Sample construction rules in `negative_price_experiments/data.py`:

- Build rolling historical windows of length `window_hours`
- Predict one future target at `horizon_hours`
- Drop samples whose target price is missing
- For selected numeric inputs, create missing indicators first
- Then apply country-local `ffill(limit=3)`
- If any selected input feature is still missing inside the input window after forward fill, drop the sample

Important:

- Missing-value masks are part of the model input
- The original target price is preserved separately so the label is never generated from filled values

## Feature groups

Public numeric features:

- `price`
- `load`
- `temp_2m_c`
- `wind_speed_10m_ms`
- `shortwave_radiation_wm2`
- `cloud_cover_pct`
- `precipitation_mm`
- `pressure_msl_hpa`

Renewables add:

- `Wind Onshore`
- `Solar`

Flow features add:

- `total export`
- `total import`

Calendar and time context:

- `sin_hour`
- `cos_hour`
- `sin_month`
- `cos_month`
- `is_weekend_local`
- `is_holiday_local`

Country handling:

- `E1-E5,E7-E9` tabular models use country one-hot
- `E1-E5,E10` sequence models use country embedding
- planned `E11-E16` sequence experiments should also use country embedding unless a later design explicitly changes that
- `E6` disables country features and country embedding on purpose so the encoder can transfer to unseen target markets

## Split protocol

Default walk-forward validation folds for `E1-E5` and `E7-E10` are defined in `WALK_FORWARD_FOLDS`.
Planned `E11-E16` should follow the same target-time-based fold protocol unless there is an explicitly documented ablation.
The final retraining and final test windows are defined in:

- `FINAL_TRAIN_RANGE`
- `FINAL_TEST_RANGE`

This is the key leakage guard:

- fold membership is based on `target_time`

Do not change this to `anchor_time`.

## Model behavior

Implemented tabular models:

- `Majority`
- `LogisticRegression`
- `XGBoost`
- `LightGBM`
- `CatBoost`
- `XGBoostWeightedCalibrated`

Implemented sequence models:

- `GRU`
- `TCN`
- `PatchTST`

Calibration notes for `XGBoostWeightedCalibrated`:

- The base learner uses class-weighted XGBoost on a leakage-safe train subset
- Probability calibration is fitted on a later holdout slice from the training window only
- The calibrator currently supports `sigmoid` or `isotonic` mapping

Sequence model implementation notes:

- Training is CPU-compatible
- Validation model selection uses best checkpoint by validation PR-AUC
- Final full-train models are retrained from scratch using the selected epoch count
- the current default sequence setup uses a `72` hour historical window in implemented configs
- planned `E11-E13` explicitly test longer windows so the model can observe more historical hours before `t+h`

Robustness behavior:

- If a training split contains only one class, the pipeline falls back to a constant-probability predictor for that split instead of crashing

## E6 transfer-learning protocol

`E6` is implemented in `run_transfer_experiment(...)`.

Source phase:

- Pretrain GRU on source markets
- Select source checkpoint on source validation PR-AUC

Target phase:

- Evaluate `ZeroShot`
- Evaluate `TargetOnly`
- Evaluate `TransferFineTune`

Budgets are defined by `AdaptBudget` ranges inside `TransferConfig`.

Scaling rules:

- `ZeroShot` and `TransferFineTune` use the source-fitted scaler
- `TargetOnly` uses the target-train-fitted scaler

## Output artifacts

Each experiment directory writes:

- `sample_manifest.csv`
- `metrics_summary.csv`
- `predictions.csv`

If predictions exist, the runner also writes:

- `country_metrics.csv`
- `monthly_metrics.csv`

Meaning:

- `sample_manifest.csv`: every valid sample after preprocessing and leakage-safe filtering
- `metrics_summary.csv`: fold-level and final metrics
- `predictions.csv`: row-level probabilities with metadata
- `country_metrics.csv`: grouped metrics by country
- `monthly_metrics.csv`: grouped metrics by month

## Testing and validation

Run all tests:

```bash
python3 -m pytest -q
```

Relevant tests:

- `tests/test_negative_price_experiments_data.py`
- `tests/test_negative_price_experiments_runner.py`

These tests cover:

- target-time-based sample selection
- missing-value and drop rules
- artifact generation
- dependency-aware runner behavior

## Extension rules for future agents

If you modify this framework, keep these rules:

- Do not change the task into price regression
- Do not assign folds by anchor time
- Do not fit scalers or threshold selection on validation or test data
- Do not remove missing-value masks from model inputs without a clear replacement
- Do not add country embedding to `E6`

Preferred extension order:

- add more reporting first
- add ablations second
- add new sequence encoders third
- only then add more advanced transfer variants

For the current roadmap:

- implement `E11-E13` first
- review their validation and test behavior
- only then lock the exact design of `E14-E16`

## Practical note on file placement

This guide is intentionally separate from `AGENTS.md`.
`AGENTS.md` should stay short and stable as a project-level instruction file.
Detailed implementation guidance belongs here so future agents can read it without turning `AGENTS.md` into a long mixed document.
