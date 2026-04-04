# 6002 Project Working Brief

## Project title
System Conditions Behind Negative Electricity Prices in High-Renewable European Power Markets: A Deep Learning-Based Event Prediction Study

## Core objective
This project should study and predict future negative electricity price events in European power markets using multivariate time-series data. The task is not ordinary price-value forecasting. It should be framed as future event classification based on past and current system conditions.

## Scope aligned with the proposal
- Learn the system operating conditions associated with negative electricity prices.
- Use electricity price data to construct binary labels.
- Use generation, demand, cross-border flow, weather, and calendar or institutional variables as predictors when available.
- Emphasize representation learning from multivariate temporal data rather than manual feature design alone.

## Professor feedback translated into requirements
- Focus on predicting negative electricity price events, not only describing them.
- Define the task with a clear time index.
- Inputs must only contain information available up to time `t`.
- Outputs must refer to a future observation or future time window, such as `t + h`.
- Avoid any leakage between input features and future labels.
- Treat the modeling pipeline generically as a temporal encoder plus classifier.
- Do not assume in advance that one model family is optimal.
- The final layer should output a binary class or event probability for a future negative-price event.

## Fixed project definition
- Build samples from rolling historical windows, for example the previous 24 hours.
- Predict whether a negative price event will occur at a defined future horizon.
- State the label definition explicitly, for example:
  `y_(t+h) = 1` if the electricity price at the future target hour is below zero, otherwise `0`.
- If a future window is used instead of a single hour, define the event rule clearly.
- The main task must remain future event classification, not same-time description and not plain price regression.
- Every dataset split, feature pipeline, and experiment must preserve temporal causality.

## Recommended research questions
1. Under what combinations of generation mix, demand level, cross-border flow, and weather conditions do future negative price events occur?
2. Can deep learning models learn latent system-state representations that improve future event prediction?
3. Do learned temporal representations outperform manually engineered features for discriminating future negative price events?

## Modeling principles
- The fixed part of the study is the problem definition, not a specific architecture.
- Preferred framing:
  past window -> temporal encoder -> classifier -> future negative-price event probability
- Start with leakage-free classification baselines before adding model complexity.
- Compare candidate sequence encoders only after the forecasting setup, labels, and time indexing are fixed.

## Flexible model choices
- LSTM or GRU, TCN, Transformer-based encoders, or similar sequence models are acceptable candidates.
- The final model should be selected by validation performance, robustness, and consistency with the forecasting setup.
- Do not claim any model family is optimal before empirical comparison.

## Evaluation
- Use classification metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- Pay attention to class imbalance because negative price events may be relatively rare.
- If possible, report results by forecast horizon and by market or country group.

## Priority tasks for future work
1. Finalize the exact forecast horizon and label definition.
2. Build a leakage-free supervised dataset with explicit input and output timestamps.
3. Train and compare baseline and deep learning classifiers.
4. Interpret which system conditions are associated with elevated future negative-price risk.
5. Keep all writing and experiments consistent with the event-prediction framing above.

## What to avoid
- Do not present the project as same-time classification without forecasting logic.
- Do not mix future information into the feature window.
- Do not describe the main task as plain electricity price regression.
- Do not lock the study to one model family before the target definition and time indexing are clear.
- Do not describe a candidate architecture as the guaranteed best method without experimental evidence.

## Usage note
Any future conversation, revision, analysis, or code change in this directory should follow this brief unless the user explicitly changes the project scope.

## Code guide
For experiment implementation details, code entrypoints, module responsibilities, artifact outputs, and extension rules, read [EXPERIMENT_CODE_GUIDE.md](./EXPERIMENT_CODE_GUIDE.md) before changing the experiment pipeline.

## Literature experiment guide
For literature-informed experiment design guidance, read [LITERATURE_EXPERIMENT_GUIDE.md](./LITERATURE_EXPERIMENT_GUIDE.md) before proposing new experiment families, changing the main modeling direction, or justifying why a new experiment is worth running.

This document exists to convert the related literature into practical experiment decisions for this repository. It should be used to keep future experiment planning aligned with:
- the project's event-prediction framing rather than plain price regression,
- literature-supported priorities such as calibration, mechanism-aware feature design, and benchmark discipline,
- and a defensible experiment order for future work after the current `E23/E25-E28` line.

## Direction log
The latest experiment-direction judgment should be recorded in [EXPERIMENT_DIRECTION_NOTES.md](./EXPERIMENT_DIRECTION_NOTES.md).

If a future conversation produces a materially updated view on:
- which backbone should remain the main deep-learning line,
- which experiment directions are feasible or not feasible,
- what the recommended next experiment order should be,
- or when a deferred direction such as hybrid modeling becomes justified,

then synchronize that new judgment into `EXPERIMENT_DIRECTION_NOTES.md` so other threads can reuse the latest roadmap.

## Experiment planning workflow
For newly proposed experiment families, keep the workflow in this order by default:

1. write down the experiment design and comparison logic first,
2. confirm the plan against the latest completed results,
3. only then implement the new experiment IDs in code.

Do not preemptively implement a new experiment family before its design has been written down, unless the user explicitly asks to skip the design-first step.

## Experiment launch convention
For future training-script launch commands, use the following shell format by default:

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
python run_negative_price_experiments.py \
  --data-file ALL_COUNTRIES_2024_2025_WITH_ERA5.csv \
  --output-dir "outputs/experiment_outputs_${RUN_ID}" \
  --experiments <EXPERIMENT_IDS> \
  --skip-unavailable-models
```

Notes:
- Keep the `RUN_ID=$(date +%Y%m%d_%H%M%S)` line at the top of the launch snippet.
- Keep the command in multi-line backslash format.
- Replace `<EXPERIMENT_IDS>` with the required comma-separated experiment IDs.
- Write outputs under the `outputs/` folder using the timestamped directory pattern `outputs/experiment_outputs_${RUN_ID}`.
- Future responses that provide experiment start commands should follow this format unless the user explicitly requests a different one.
