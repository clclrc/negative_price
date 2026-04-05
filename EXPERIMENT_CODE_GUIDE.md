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

Extended experiment IDs:

- `E11`: main 20-market public-feature deep-learning experiment with `GRU`, `window=120`, `h=6`
- `E12`: main 20-market public-feature deep-learning experiment with `GRU`, `window=168`, `h=6`
- `E13`: main 20-market public-feature deep-learning experiment with `TCN`, `window=168`, `h=6`
- `E14`: main 20-market public-feature `GRU` experiment with focal loss, `window=168`, `h=6`
- `E15A`: 15-market public-feature `GRU` experiment, `window=168`, `h=6`
- `E15B`: 15-market renewables-feature `GRU` experiment, `window=168`, `h=6`
- `E16A`: 7-market public-feature `GRU` experiment, `window=168`, `h=6`
- `E16B`: 7-market cross-border-flow `GRU` experiment, `window=168`, `h=6`
- `E17A`: 15-market public-feature `GRU` experiment on the renewables-shared valid-sample subset, `window=168`, `h=6`
- `E17B`: 15-market renewables-feature `GRU` experiment on the same renewables-shared valid-sample subset, `window=168`, `h=6`
- `E18`: repeated-seed version of `E17B` with aggregated metrics across multiple random seeds
- `E19`: 15-market renewables-feature `GRUHybrid` experiment that fuses sequence representations with handcrafted tabular features
- `E20`: 20-market renewables-feature `GRU` experiment with missingness-aware window retention, `window=168`, `h=6`
- `E21`: repeated-seed version of `E19` with aggregated metrics across multiple random seeds
- `E22A`: 15-market public-feature `GRUHybrid` experiment on the renewables-shared valid-sample subset, `window=168`, `h=6`
- `E22B`: 15-market renewables-feature `GRUHybrid` experiment on the same shared valid-sample subset, `window=168`, `h=6`
- `E23`: 20-market public-feature `GRUHybrid` experiment, `window=168`, `h=6`
- `E24`: 20-market renewables-feature `GRUHybrid` experiment with missingness-aware window retention, `window=168`, `h=6`
- `E25`: repeated-seed version of `E23` with aggregated metrics across multiple random seeds
- `E26`: 20-market public-feature `GRUHybrid` experiment with focal loss, `window=168`, `h=6`
- `E27`: 20-market public-feature `GRUHybrid` experiment with a larger sequence training budget, `window=168`, `h=6`
- `E28`: 20-market renewables-feature `GRUHybrid` experiment with missingness-aware window retention and a larger sequence training budget, `window=168`, `h=6`
- `E29`: 20-market public-feature `GRUHybridAttn` experiment with temporal attention pooling, `window=168`, `h=6`
- `E30`: 20-market public-feature `GRUHybridGated` experiment with gated sequence-tabular fusion, `window=168`, `h=6`
- `E31`: 20-market public-feature `GRUHybridGated` experiment with mechanism-aware engineered tabular features, `window=168`, `h=6`
- `E32`: 20-market public-feature `GRUHybridGatedMultiTask` experiment with mechanism-aware features and an auxiliary future-price target, `window=168`, `h=6`
- `E33`: repeated-seed version of `E30` with aggregated metrics across multiple random seeds
- `E34`: repeated-seed version of `E31` with aggregated metrics across multiple random seeds
- `E35`: validation-weighted late-fusion ensemble wrapper over `E30` and `E31`
- `E36`: calibration wrapper that selects the strongest validation-time candidate among `E30`, `E31`, and `E35`
- `E37`: stability-tuned `E30` with a lower sequence learning rate and a larger training budget
- `E38`: stability-tuned `E31` with the same stability-oriented training changes
- `E39`: stacking ensemble wrapper over `E30` and `E31` using validation-time member predictions as stacker inputs
- `E40`: cross-seed ensemble wrapper over `E33` and `E34` that turns repeated-seed diversity into weighted ensemble members
- `E41`: 20-market public-feature `LogisticRegression` baseline under the current main task setup, `window=168`, `h=6`
- `E42`: 20-market public-feature `XGBoost` baseline under the current main task setup, `window=168`, `h=6`
- `E43`: 20-market public-feature `CatBoost` baseline under the current main task setup, `window=168`, `h=6`
- `E44`: 20-market public-feature `LightGBM` baseline under the current main task setup, `window=168`, `h=6`
- `E45`: 20-market public-feature `GRUMultiMarket` experiment that jointly encodes all markets at the same anchor time, `window=168`, `h=6`
- `E46`: 20-market public-feature `GraphTemporal` experiment that adds a dense market-interaction layer on top of joint multi-market encodings, `window=168`, `h=6`
- `E47`: 20-market public-feature `GraphTemporalHybrid` experiment that fuses graph-temporal market representations with mechanism-aware tabular features, `window=168`, `h=6`
- `E48`: repeated-seed version of `E47` with aggregated metrics across multiple random seeds

Important:

- `E11`, `E12`, `E13`, `E14`, `E15A`, `E15B`, `E16A`, `E16B`, `E17A`, `E17B`, `E18`, `E19`, `E20`, `E21`, `E22A`, `E22B`, `E23`, `E24`, `E25`, `E26`, `E27`, `E28`, `E29`, `E30`, `E31`, `E32`, `E33`, `E34`, `E35`, `E36`, `E37`, `E38`, `E39`, `E40`, `E41`, `E42`, `E43`, `E44`, `E45`, `E46`, `E47`, and `E48` listed above are implemented config defaults
- `E14` is the implemented imbalance-aware extension of the current `E12`-style deep backbone
- `E15A/E15B` form a paired renewables-track comparison on the same 15-country subset
- `E16A/E16B` form a paired flow-track comparison on the same 7-country subset
- `E17A/E17B` form the stricter renewables-track comparison on a shared valid-sample subset
- `E18` wraps repeated random seeds internally and writes both raw and aggregated metrics
- `E19` is the first implemented hybrid deep-learning experiment in this repository
- `E20` keeps sequence windows with missing renewable inputs and relies on missingness masks instead of dropping those samples
- `E21` extends the hybrid line with repeated-seed stability reporting
- `E22A/E22B` let the repository compare hybrid `public` versus hybrid `renewables` under the same shared valid-sample subset
- `E23` tests whether the hybrid architecture itself helps on the full `20-country` main setup
- `E24` is the current full-setup renewables-aware hybrid extension
- `E25` is the repeated-seed stability check for the `E23` mainline
- `E26` tests whether the full-setup hybrid line benefits from focal loss
- `E27` tests whether the full-setup hybrid line benefits from a larger training budget
- `E28` revisits the full-setup renewables-aware hybrid line with both missingness-aware retention and a larger training budget
- `E29` tests whether temporal attention pooling helps the `E23`-style hybrid line under the same full setup
- `E30` tests whether gated fusion between sequence and tabular representations helps more than simple concatenation
- `E31` keeps the `E30`-style gated hybrid but adds mechanism-aware engineered features such as ramps, drawdowns, and calendar interactions
- `E32` adds a future-price auxiliary target to the `E31`-style mainline while keeping future-event classification as the primary output
- `E33` is the repeated-seed stability check for the `E30` mainline
- `E34` is the repeated-seed stability check for the `E31` mechanism-aware branch
- `E35` trains `E30` and `E31` as members and then builds a validation-`PR-AUC`-weighted late-fusion ensemble from their predictions
- `E36` trains or reuses `E30`, `E31`, and `E35`, selects the strongest validation-time candidate, and applies probability calibration to that selected branch
- `E37` keeps the `E30` architecture fixed but lowers the learning rate while extending the epoch and patience budget to reduce optimization variance
- `E38` applies the same stability-oriented training changes to the `E31` mechanism-aware branch
- `E39` trains `E30` and `E31` as members and fits a small logistic-regression stacker on their validation-time probabilities
- `E40` trains or reuses `E33` and `E34`, expands their raw seed members, and builds a validation-`PR-AUC`-weighted cross-seed ensemble
- `E41-E44` are matched classical baselines for the current main task, using the same `20-country + public + window=168 + h=6` problem definition with tabular lag and rolling-stat features
- `E45` is the first implemented multi-market deep baseline; it keeps a shared GRU encoder but stops treating each market as an isolated single-sequence problem
- `E46` keeps the same multi-market data path as `E45` but adds a market-interaction layer to model cross-market state propagation
- `E47` adds a gated fusion branch that combines graph-temporal market representations with the repository's strongest mechanism-aware tabular summaries
- `E48` is the stability check for that graph-temporal hybrid line

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

Current next deep-learning phase:

- `E23` is the current main deep-learning benchmark for the full `20-country` setup
- use `E25` to measure seed stability for that full-setup hybrid benchmark
- use `E26` to test whether focal loss helps the `E23` line
- use `E27` to test whether a larger training budget helps the `E23` line
- keep `E24` and `E28` as the conditional renewables-aware full-setup branch rather than the default mainline
- keep `E21` and `E22A/E22B` as supporting evidence that the hybrid and renewables directions were real before the mainline switched to `E23`
- once the `E29-E32` branch is judged, use `E33-E36` as the stability, complementarity, and calibration branch around `E30/E31` rather than reopening older backbone questions

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
- `E17A/E17B` can override which feature group defines valid windows through a shared sample-filter feature group
- `E20`, `E24`, and `E28` can preserve windows with missing renewable inputs and expose that missingness through mask channels instead of dropping the sample

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
- `E11-E14,E15A,E15B,E16A,E16B,E17A,E17B,E18,E19,E20,E21,E22A,E22B,E23,E24,E25,E26,E27,E28,E29,E30,E31,E32` sequence experiments also use country embedding
- `E6` disables country features and country embedding on purpose so the encoder can transfer to unseen target markets

## Split protocol

Default walk-forward validation folds for `E1-E5` and `E7-E10` are defined in `WALK_FORWARD_FOLDS`.
`E11-E14,E15A,E15B,E16A,E16B,E17A,E17B,E18,E19,E20,E21,E22A,E22B,E23,E24,E25,E26,E27,E28,E29,E30,E31,E32` follow the same target-time-based fold protocol.
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
- `GRUHybrid`
- `GRUHybridAttn`
- `GRUHybridGated`
- `GRUHybridGatedMultiTask`
- `GRUMultiMarket`
- `GraphTemporal`
- `GraphTemporalHybrid`
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
- the original sequence baselines use a `72` hour historical window
- `E11-E14,E15A,E15B,E16A,E16B,E17A,E17B,E18,E19,E20,E21,E22A,E22B,E23,E24,E25,E26,E27,E28,E29,E30,E31,E32` explicitly test longer windows so the model can observe more historical hours before `t+h`

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

- keep `E23` as the main full-setup deep-learning baseline
- use `E25` to measure whether `E23` is stable across random seeds
- use `E26` and `E27` as the first architecture-preserving follow-ups on that same line
- treat `E24` and `E28` as the conditional full-setup renewables-aware branch
- keep `E21` and `E22A/E22B` as supporting evidence for why the project moved from pure `GRU` models to the hybrid mainline
- use `E29` and `E30` as the first architectural follow-ups once the `E23` line is stable enough
- use `E31` to test mechanism-aware engineered features without changing the main task definition
- use `E32` to test multi-task event-plus-price learning only after the stronger single-task hybrids are available
- use `E33` and `E34` to check whether the `E30/E31` gains survive seed variation
- use `E35` to test whether `E30` and `E31` are complementary enough to beat either single branch
- use `E36` to test whether the strongest available completed branch can be made more decision-useful through calibration

## Practical note on file placement

This guide is intentionally separate from `AGENTS.md`.
`AGENTS.md` should stay short and stable as a project-level instruction file.
Detailed implementation guidance belongs here so future agents can read it without turning `AGENTS.md` into a long mixed document.
