# Literature-Informed Experiment Design Guide

## Purpose
This note summarizes what the recent literature suggests for future experiments in this repository.

The goal is not to collect every paper on electricity prices. The goal is to extract the parts that should change how new experiments are designed for future negative-price event prediction.

## Executive judgment

- Direct forecasting of future negative-price events does exist, but it is still much less common than ordinary electricity price regression.
- Most related papers fall into one of four buckets:
  `driver analysis`, `threshold-event classification`, `price/distribution forecasting`, or `generic deep-learning price prediction`.
- This project's clearest research niche remains:
  `past window -> temporal encoder -> classifier -> future negative-price event probability`
  under a leakage-free multi-market European setup.
- Therefore, the best future experiments are the ones that improve:
  `task design`, `probability quality`, `mechanism alignment`, and `generalization`.
- The best future experiments are not the ones that switch to a more fashionable backbone without a clear signal that the simpler line is exhausted.

## What the literature says and what it implies

### 1. Negative-price studies often start with simple event models

Representative papers:

- Aust and Horsch (2020), ["Negative market prices on power exchanges: Evidence and policy implications from Germany"](https://doi.org/10.1016/j.tej.2020.106716)
- Biber et al. (2022), ["Negative price spiral caused by renewables? Electricity price prediction on the German market for 2030"](https://doi.org/10.1016/j.tej.2022.107188)
- Liu et al. (2022), ["Forecasting the occurrence of extreme electricity prices using a multivariate logistic regression model"](https://doi.org/10.1016/j.energy.2022.123417)

Main pattern:

- These papers often use `logit` or `logistic regression`.
- Inputs are mechanism-facing variables such as past prices, demand, renewable output, reserve margin, interconnector flow, fuel and carbon variables, and time indicators.
- The output is not the future price value itself, but the probability that an extreme or negative-price event occurs.

Implications for this repository:

- Simple probabilistic classifiers are not weak baselines. They are literature-aligned baselines.
- Any new deep-learning experiment should still be compared against strong simple baselines such as `logistic regression`, `CatBoost`, and `LightGBM`.
- Interpretable event models should remain part of the experiment story because they help explain which system states matter.

### 2. Many papers still solve the easier problem: price forecasting, not event forecasting

Representative papers:

- Lago et al. (2018), ["Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms"](https://doi.org/10.1016/j.apenergy.2018.02.069)
- Lago et al. (2021), ["Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark"](https://doi.org/10.1016/j.apenergy.2021.116983)
- Li and Becker (2021), ["Day-ahead electricity price prediction applying hybrid models of LSTM-based deep learning methods and feature selection algorithms under consideration of market coupling"](https://arxiv.org/abs/2101.05249)
- Failing et al. (2026), ["Deep learning-based prediction models for spot electricity market prices in the Spanish market"](https://doi.org/10.1016/j.matcom.2025.07.010)

Main pattern:

- The literature is still dominated by point price forecasting.
- Deep models are usually trained to predict future prices directly.
- Exogenous variables matter a lot: load, renewable generation, weather, fuel prices, and market-coupling information repeatedly appear.

Implications for this repository:

- Your project should keep emphasizing that it is not plain price regression.
- But the feature choices in the price-forecasting literature are still useful because they reveal which future-event predictors are plausible.
- Market-coupling and exogenous system-state variables should remain central in feature design.

### 3. Probabilistic price forecasting is especially relevant for your next stage

Representative papers:

- Marcjasz et al. (2023), ["Distributional neural networks for electricity price forecasting"](https://doi.org/10.1016/j.eneco.2023.106843)
- Maciejowska et al. (2021), ["Enhancing load, wind and solar generation for day-ahead forecasting of electricity prices"](https://doi.org/10.1016/j.eneco.2021.105273)
- Ziel and Weron (2018), ["Day-ahead electricity price forecasting with high-dimensional structures: Univariate vs. multivariate modeling frameworks"](https://doi.org/10.1016/j.eneco.2017.12.016)

Main pattern:

- Some strong papers focus on predicting a full price distribution or a richer probabilistic object rather than a single point estimate.
- This matters because the downstream quantity of interest can be transformed into event risk, for example `P(price < 0)`.
- Forecast combinations and hybrid structures often improve robustness more than aggressive architectural novelty.

Implications for this repository:

- Calibration should become a first-class evaluation target.
- A secondary research line can predict future price distributions and derive `P(price < 0)` from them.
- This should remain a secondary line, not replace the main event-classification framing.

### 4. The literature is expanding toward deeper models, but evaluation quality is uneven

Representative papers:

- Jasiński (2025), ["A Review of Recent Trends in Electricity Price Forecasting Using Deep Learning Techniques"](https://doi.org/10.3390/en18246422)

Main pattern:

- `CNN`, `LSTM`, `GRU`, and hybrid models remain common.
- `Transformer` use is increasing.
- Many recent papers still use short or weak evaluation windows, which makes bold architecture claims less credible.

Implications for this repository:

- Do not change the mainline backbone just because newer papers use transformers.
- Long chronological evaluation, repeated seeds, and stable benchmark comparisons are part of the contribution.
- This supports the current repo direction of keeping a strong tree baseline and a stable deep-learning mainline rather than chasing architecture churn.

### 5. Direct negative-price forecasting papers suggest a practical two-track idea

Representative paper:

- Loizidis et al. (2025), ["Integrating PNN classification and ELM-Bootstrap for enhanced Day-Ahead negative price forecasting"](https://doi.org/10.1016/j.apenergy.2025.126013)

Main pattern:

- The paper explicitly treats negative prices as a special forecasting target.
- It combines classification with a second price-forecasting component.
- It also uses asymmetric misclassification logic motivated by market costs.

Implications for this repository:

- A cost-sensitive branch is legitimate, but it should be framed as an application variant rather than the main scientific metric.
- A multi-task branch that predicts both `future event probability` and a `future price statistic` is now well justified.

## Design principles for future experiments

### 1. Keep the task definition strict

- Inputs must only use information available up to anchor time `t`.
- Labels must refer to a future target time or future target window.
- Split by `target_time`, not `anchor_time`.
- Do not let derived features sneak future information into the input window.

### 2. Preserve the current benchmark discipline

- Keep strong tabular baselines in every serious comparison.
- Treat `CatBoost` and the better tree baselines as score references, not optional baselines.
- Treat the current `GRUHybrid` line as the main deep-learning reference unless new evidence clearly overturns it.

### 3. Favor mechanism-aligned feature additions over random architecture expansion

Highest-value feature directions from the literature:

- renewable output and renewable share
- demand level and demand ramps
- residual load or net load proxies
- net import, net export, and congestion-related flow proxies
- calendar and low-demand context
- weather variables that are causal drivers of renewable output or demand
- fuel and carbon variables if a clean causal data pipeline becomes available

### 4. Evaluate probabilities, not only classifications

- Keep `PR-AUC` as the main ranking metric.
- Add `Brier score` and calibration analysis for any new probability-focused experiment.
- Keep `F1`, `precision`, `recall`, `ROC-AUC`, and balanced accuracy as supporting metrics.
- For event-risk use cases, a well-calibrated `0.18` can be more useful than an overconfident but miscalibrated `0.70`.

### 5. Separate scientific questions cleanly

Avoid experiments that change all of these at once:

- feature space
- label definition
- horizon
- backbone
- loss function
- thresholding rule

The literature is full of hard-to-interpret hybrid jumps. This project should stay cleaner than that.

## What this means for the current repository

### Mainline judgment

- Keep `E23` as the main deep-learning benchmark.
- Keep `E25-E27` as the immediate stabilization path.
- Keep `E24/E28` as conditional branches, not the default next move.
- Keep the `flow` direction secondary until missingness and comparison quality improve.

### Why this is still the right mainline

- Recent literature does not show a decisive reason to abandon `GRUHybrid` for a transformer-first strategy.
- The literature repeatedly rewards careful feature design, probability handling, and benchmark rigor.
- That matches the strongest signals already present in this repo.

## Recommended next experiment families after `E25-E27`

### Family N1: Probability calibration branch

Question:

- Are the current event probabilities well calibrated enough for risk-oriented interpretation?

Design:

- Take the strongest current classifier and add calibration evaluation.
- Compare uncalibrated outputs against temperature scaling, Platt scaling, or isotonic calibration fitted only on validation data.

Why this is justified:

- The literature on probabilistic EPF shows that probability quality matters, not just ranking.

What to report:

- `PR-AUC`
- `Brier score`
- reliability plots
- threshold sensitivity

### Family N2: Multi-horizon event prediction

Question:

- Does a shared encoder help when forecasting `h = 6`, `12`, and `24` together?

Design:

- Keep the same causal input window.
- Compare independent single-horizon heads against a shared encoder with multiple horizon-specific outputs.

Why this is justified:

- Day-ahead forecasting papers often exploit structure across hours.
- Your project can adapt that idea to future event classification.

What to report:

- per-horizon `PR-AUC`
- average rank across horizons
- whether shorter-horizon gains trade off against longer-horizon degradation

### Family N3: Future-window event labels

Question:

- Is the model better at predicting whether any negative-price event occurs within a future window than at predicting a single future hour?

Design:

- Keep the current single-hour label as the anchor baseline.
- Add explicit alternatives such as:
  `any negative price in [t+1, t+6]`
  or
  `at least 2 negative hours in [t+1, t+6]`.

Why this is justified:

- Many practical decisions care about near-term risk windows, not one hour in isolation.
- This stays faithful to the event-prediction framing.

Risk:

- A wider window changes class balance and may make the task easier.

### Family N4: Mechanism-aware hybrid feature branch

Question:

- Can simple causal summary variables help the current hybrid mainline more than raw feature expansion alone?

Candidate engineered features:

- residual load proxy
- renewable share
- rolling ramps in load, wind, and solar
- recent price volatility and recent price minimum
- net import minus export
- holiday-night interaction flags

Why this is justified:

- The literature repeatedly shows that system-balance variables matter.
- These features are also easier to interpret than hidden states alone.

Rule:

- Only include features computable from information available by time `t`.

### Family N5: Multi-task event plus price branch

Question:

- Does a shared encoder learn a better system-state representation when asked to predict both the future event and a future price target?

Design:

- Main output:
  future negative-price event probability
- Auxiliary output:
  future price value, price quantile, or price sign/margin

Why this is justified:

- It reuses the much larger price-forecasting literature without changing the main task definition.
- It is also the cleanest way to borrow information from price regression while keeping event classification primary.

### Family N6: Transfer and market-generalization branch

Question:

- Does the learned encoder generalize across countries and regimes?

Design:

- Reuse the transfer logic already present in `E6`-style setups.
- Test a stronger encoder under held-out-country or leave-some-markets-out conditions.
- If possible, test regime transfer across lower-volatility and higher-volatility subperiods.

Why this is justified:

- Market coupling papers suggest transferable cross-market structure exists.
- A multi-market contribution is stronger when it shows generalization, not only pooled fit.

### Family N7: Cost-sensitive decision branch

Question:

- Should thresholding or training weights reflect asymmetric economic costs?

Design:

- Keep this as an application branch after probability calibration is understood.
- Compare standard loss against cost-sensitive loss or threshold selection.

Why this is justified:

- Loizidis et al. explicitly motivate asymmetric costs.

Important limitation:

- This branch is useful for practice, but it should not replace the main metric-driven scientific comparison.

## Lower-priority directions

- Do not prioritize a new transformer family unless the current hybrid line clearly saturates.
- Do not reopen the flow-heavy branch before cleaning missingness and enforcing fair matched-sample comparisons.
- Do not drift into regression-only work as the main contribution.
- Do not compare experiments across different country subsets as if they were directly score-comparable.

## Practical experiment order

1. Finish `E25-E27` and confirm the stability of the current hybrid mainline.
2. Add `N1` probability calibration on the strongest current model.
3. Add `N2` multi-horizon event prediction.
4. Add `N3` future-window event labels.
5. Add `N4` mechanism-aware engineered features into the hybrid line.
6. Add `N5` multi-task event plus price prediction.
7. Add `N6` transfer and market-generalization tests.
8. Add `N7` cost-sensitive decision variants only after the probability outputs are trustworthy.

## Selected references

- Aust, B., and Horsch, A. (2020). ["Negative market prices on power exchanges: Evidence and policy implications from Germany"](https://doi.org/10.1016/j.tej.2020.106716).
- Biber, A., Tunçinan, M., Wieland, C., and Spliethoff, H. (2022). ["Negative price spiral caused by renewables? Electricity price prediction on the German market for 2030"](https://doi.org/10.1016/j.tej.2022.107188).
- Failing, J. M., Segarra-Tamarit, J., Cardo-Miota, J., and Beltran, H. (2026). ["Deep learning-based prediction models for spot electricity market prices in the Spanish market"](https://doi.org/10.1016/j.matcom.2025.07.010).
- Jasiński, T. (2025). ["A Review of Recent Trends in Electricity Price Forecasting Using Deep Learning Techniques"](https://doi.org/10.3390/en18246422).
- Lago, J., De Ridder, F., and De Schutter, B. (2018). ["Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms"](https://doi.org/10.1016/j.apenergy.2018.02.069).
- Lago, J., Marcjasz, G., De Schutter, B., and Weron, R. (2021). ["Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark"](https://doi.org/10.1016/j.apenergy.2021.116983).
- Li, W., and Becker, D. M. (2021). ["Day-ahead electricity price prediction applying hybrid models of LSTM-based deep learning methods and feature selection algorithms under consideration of market coupling"](https://arxiv.org/abs/2101.05249).
- Liu, L., Bai, F., Su, C., Ma, C., Yan, R., Li, H., Sun, Q., and Wennersten, R. (2022). ["Forecasting the occurrence of extreme electricity prices using a multivariate logistic regression model"](https://doi.org/10.1016/j.energy.2022.123417).
- Loizidis, S., Venizelou, V., Kyprianou, A., and Georghiou, G. E. (2025). ["Integrating PNN classification and ELM-Bootstrap for enhanced Day-Ahead negative price forecasting"](https://doi.org/10.1016/j.apenergy.2025.126013).
- Maciejowska, K., Nitka, W., and Weron, T. (2021). ["Enhancing load, wind and solar generation for day-ahead forecasting of electricity prices"](https://doi.org/10.1016/j.eneco.2021.105273).
- Marcjasz, G., Narajewski, M., Weron, R., and Ziel, F. (2023). ["Distributional neural networks for electricity price forecasting"](https://doi.org/10.1016/j.eneco.2023.106843).
- Prokhorov, O., and Dreisbach, D. (2022). ["The impact of renewables on the incidents of negative prices in the energy spot markets"](https://doi.org/10.1016/j.enpol.2022.113073).
- Ziel, F., and Weron, R. (2018). ["Day-ahead electricity price forecasting with high-dimensional structures: Univariate vs. multivariate modeling frameworks"](https://doi.org/10.1016/j.eneco.2017.12.016).
