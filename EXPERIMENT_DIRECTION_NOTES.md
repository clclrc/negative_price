# Experiment Direction Notes

## Purpose
This file records the current judgment on experiment directions, feasibility, risks, and recommended next steps.

It is intended to help future threads and future agents avoid re-deciding the same roadmap from scratch.

## Current status

### Best current deep-learning baseline
- `E12`: `GRU`, `window=168`, `h=6`, `20-market`, `public features`
- Current role:
  best-performing deep-learning backbone under the main forecasting setup

### High-level conclusion so far
- Longer history helps `GRU`
- `GRU` currently looks more promising than `TCN` and clearly more promising than the current `PatchTST`
- Deep learning still underperforms the strongest tree-model baselines
- The most defensible next step is to continue from `E12` rather than switching backbone again

## Observed signals from completed experiments

### Signal group 1: tree-model benchmark results (`E7-E10` comparison context)
- `E8` (`CatBoost`) is the strongest current overall benchmark in the main `20-market + public + h=6` setting
- `E7` (`LightGBM`) is competitive but not better than `E8`
- `E9` (weighted and calibrated `XGBoost`) did not improve on the stronger existing tree baselines
- `E10` (`PatchTST`) underperformed clearly and is not the preferred deep-learning direction right now

#### Direction implication
- The project should keep a strong tree-model benchmark, especially `E8`, as the main score reference
- `PatchTST` should not be the priority deep-learning backbone for the next round

### Signal group 2: longer-window deep-learning results (`E11-E13`)
- `E11`: `GRU`, `window=120`, improved over the earlier `GRU` baseline in some aspects but showed a large validation-to-test drop
- `E12`: `GRU`, `window=168`, is the best current deep-learning configuration under the main setting
- `E13`: `TCN`, `window=168`, did not beat the `E12` `GRU`

#### Direction implication
- A longer history window is useful for the current `GRU` line
- `GRU + 168h` should remain the main deep-learning backbone until new evidence overturns it
- `TCN` is currently secondary rather than the lead candidate

### Signal group 3: what the current gap means
- The best current deep-learning result still remains materially below the strongest tree baseline
- This means the next round should focus on improving the current deep-learning line rather than changing backbone again
- The most justified next levers are:
  imbalance-aware training,
  richer feature groups on matched country subsets,
  and only later hybrid modeling

### Signal group 4: imbalance-aware and richer-feature follow-up results (`E14-E16`)
- `E14` (`GRU + 168h + focal loss`) did not improve over `E12` on test `PR-AUC`
- `E14` slightly improved `F1` and clearly improved `recall` and `balanced_accuracy`
- This suggests focal loss makes the current deep-learning model more aggressive at catching events, but does not improve ranking quality under the main metric
- `E15B` (`15-country`, `renewables`) outperformed `E15A` (`15-country`, `public`) on test `PR-AUC`, `F1`, `recall`, and `balanced_accuracy`
- This is the clearest positive signal so far that richer renewable features can help the selected deep-learning backbone
- `E16B` (`7-country`, `flows`) underperformed `E16A` (`7-country`, `public`) on test `PR-AUC`, `precision`, and `F1`
- This means the current flow-feature implementation is not yet a stronger direction than the matched public-feature baseline
- `E16A` achieved the strongest absolute test score in this round, but it is a `7-country` subset result and should not be treated as directly comparable to the `20-country` main setup

#### Direction implication
- Keep `E12` as the main deep-learning reference for score comparison under the full `20-country` setup
- Treat `E14` as a recall-oriented variant, not a new mainline winner
- Increase confidence in the renewables track
- Decrease priority of the current flow track until there is a cleaner or stronger feature formulation
- Continue to defer hybrid modeling until the richer-feature direction is better understood

### Signal group 5: cleaner renewables validation and hybrid follow-up (`E17-E20`)
- `E17B` outperformed `E17A` on the same shared valid-sample subset, so the renewables gain is not just a sample-selection artifact
- The uplift remains meaningful but smaller than the original `E15B` versus `E15A` gap, which means part of the earlier gain did come from sample filtering
- `E18` confirmed that the `E17B`-style renewables gain is stable across random seeds, but the mean test `PR-AUC` is still only around `0.263`
- `E19` (`GRUHybrid`) improved over `E17B` on test `PR-AUC`, `precision`, and `F1`, making it the strongest current result on the matched `15-country` renewables subset
- `E20` increased sample coverage and recall in the `20-country` renewables setting, but it still underperformed `E12` on the main `PR-AUC` metric

#### Direction implication
- The renewables track is now confirmed rather than merely suggested
- Hybrid modeling is no longer a deferred idea; it has become the most promising next deep-learning line
- `E12` should remain the full `20-country` benchmark
- `E19` should become the main subset-level development line
- `E20` should be treated as an exploratory coverage-oriented branch rather than the new mainline model

## Direction assessment

### Direction A: `E14 = E12 + focal loss`

#### Why this direction is viable
- The task is highly imbalanced, with negative-price events forming a small minority class
- Current deep-learning models still show room for improvement in `PR-AUC`, `precision`, and `F1`
- This direction changes the training objective without changing the data scope, making it a clean follow-up to `E12`

#### Main research question answered
- Can imbalance-aware training improve future negative-price event detection under the same causal forecasting setup?

#### Risks
- Recall may increase while probability calibration worsens
- Precision may not improve if the model simply becomes more aggressive
- If combined with too many other changes at once, attribution becomes unclear

#### Recommendation
- High priority
- Run this before richer-feature experiments because it is the cleanest continuation of the `E12` line
- Status:
  implemented and completed

#### Observed outcome
- Test `PR-AUC` decreased from `0.2918` (`E12`) to `0.2844`
- Test `F1` changed only marginally, from `0.3273` to `0.3285`
- Test `recall` increased from `0.5911` to `0.7076`
- Test `balanced_accuracy` increased from `0.7767` to `0.8295`

#### Updated judgment
- Focal loss is useful if higher event recall is the priority
- Focal loss is not currently the best choice if `PR-AUC` remains the primary selection metric
- `E14` should be treated as a recall-oriented branch rather than replacing `E12` as the main deep-learning baseline

## Direction B: `E15A/E15B` renewables track

### `E15A`: `15-country`, `public`, `GRU`, `window=168`, `h=6`
### `E15B`: `15-country`, `renewables`, `GRU`, `window=168`, `h=6`

#### Why this direction is viable
- Renewable output is directly related to many future negative-price mechanisms
- Sequence models may benefit from the temporal dynamics of wind and solar more than tabular baselines do
- A paired design with `E15a` and `E15b` keeps the country set fixed and isolates the value of the added renewable features

#### Main research question answered
- Does a stronger deep sequence model gain predictive value from renewable-generation features beyond the public baseline?

#### Risks
- Renewable data coverage is not equally complete for all countries
- If the country set changes, comparisons against the 20-country main setup become less direct
- Gains may reflect country selection rather than feature-group value unless the baseline is matched on the same country subset

#### Recommendation
- High priority after `E14`
- Use only the countries with sufficiently complete renewable coverage
- Always compare `E15B` against `E15A`, not directly against `E12`

#### Observed outcome
- `E15B` improved test `PR-AUC` from `0.2219` to `0.2665`
- `E15B` improved test `F1` from `0.2990` to `0.3186`
- `E15B` improved test `recall` from `0.4739` to `0.6394`
- The renewables track therefore produced a positive direction signal for the selected `GRU + 168h` backbone
- However, `E15B` used fewer valid samples than `E15A` because of feature availability, so this is not yet a perfectly sample-matched comparison

#### Updated judgment
- Renewables are the strongest current richer-feature direction for deep learning in this project
- This track should be prioritized ahead of further flow-feature expansion
- A future cleaner comparison on a shared valid-sample subset would strengthen the conclusion

## Direction C: `E16A/E16B` flow track

### `E16A`: `7-country`, `public`, `GRU`, `window=168`, `h=6`
### `E16B`: `7-country`, `flows`, `GRU`, `window=168`, `h=6`

#### Why this direction is viable
- Cross-border flow variables capture system-state constraints that are highly relevant to future negative prices
- This is one of the most mechanism-aligned feature extensions in the project
- A paired design with `E16a` and `E16b` isolates the value of flow features while controlling the country subset

#### Main research question answered
- Does the selected deep sequence model gain predictive value from transmission-flow information beyond the public baseline?

#### Risks
- Flow coverage is available for fewer countries, so absolute scores are not directly comparable with 20-country experiments
- Sample size may become smaller and more regime-specific
- A strong result here may reflect an easier subset rather than a universally better model

#### Recommendation
- High priority after `E14`
- Use only the countries with sufficiently complete flow coverage
- Compare `E16B` primarily against `E16A`, then secondarily against historical flow baselines

#### Observed outcome
- `E16B` reduced test `PR-AUC` from `0.5192` to `0.3848`
- `E16B` reduced test `F1` from `0.5489` to `0.4568`
- `E16B` increased `recall` slightly, but at a meaningful cost in ranking quality and precision
- `E16A` is the strongest absolute result in this round, but that strength belongs to the `7-country public-feature` subset baseline rather than to the added flow features
- `E16B` also used fewer valid samples than `E16A`, so missingness likely contributes to the instability of the comparison

#### Updated judgment
- The current flow-feature formulation should not be treated as a successful extension
- Flow features are lower priority than renewables for the next round
- If revisited, this direction should first improve missing-data handling or enforce a shared valid-sample comparison

## Direction D: Hybrid sequence encoder + handcrafted features

### Definition
- Use a sequence encoder such as the `E12`-style `GRU` to produce a learned temporal representation
- Concatenate that learned representation with handcrafted lag/statistical tabular features before the final classifier

#### Why this direction is viable
- Current tree baselines benefit from strong handcrafted summary features
- Current deep-learning models benefit from learned temporal representations
- A hybrid model tests whether these two information sources are complementary rather than mutually exclusive

#### Main research question answered
- Do learned temporal representations and handcrafted statistical summaries provide complementary predictive information?

#### Risks
- This is not a clean continuation experiment if introduced too early
- It changes the representation, feature space, and classifier jointly
- If used before `E14-E16`, it becomes much harder to interpret whether gains come from better temporal learning or simply from adding engineered features

#### Recommendation
- This direction is now high priority
- Use `E19` as the first strong hybrid reference result
- Continue hybrid development before spending more effort on the current `E20`-style missingness-aware renewables line

#### Observed outcome
- `E19` improved test `PR-AUC` from `0.2665` (`E17B`) to `0.2775`
- `E19` improved test `F1` from `0.3186` to `0.3588`
- `E19` improved test `precision` from `0.2121` to `0.2568`
- `E19` did give up some recall relative to `E17B`, but the overall trade-off is better under the current objective
- `E19` also showed a smaller validation-to-test drop than the pure-sequence renewables line

#### Updated judgment
- Hybrid modeling has now earned follow-up investment
- The next iteration should focus on confirming and strengthening the `E19` line
- Hybrid is currently the strongest candidate for the next deep-learning mainline on the renewables subset

## Recommended order

1. Keep `E12` as the main full-setup deep-learning benchmark
2. Keep `E19` as the main renewables-subset development line
3. Run repeated-seed validation for `E19`
4. Compare hybrid `public` versus hybrid `renewables` on the same matched subset
5. Revisit a `20-country` renewables-hybrid extension only after the hybrid gains are confirmed

## Implemented next-step experiment definitions

- `E17A`: `15-country`, `public`, `GRU`, `window=168`, `h=6`, but filtered to the renewables-shared valid sample subset
- `E17B`: `15-country`, `renewables`, `GRU`, `window=168`, `h=6`, using the same renewables-shared valid sample subset
- `E18`: repeated-seed version of `E17B`, currently configured with three random seeds and root-level aggregated metrics
- `E19`: `15-country`, `renewables`, `GRUHybrid`, `window=168`, `h=6`, combining sequence representations with handcrafted tabular features
- `E20`: `20-country`, `renewables`, `GRU`, `window=168`, `h=6`, with missingness-aware window retention instead of dropping windows that still contain missing renewable inputs
- `E21`: repeated-seed version of `E19`, currently configured with three random seeds and root-level aggregated metrics
- `E22A`: `15-country`, `public`, `GRUHybrid`, `window=168`, `h=6`, using the renewables-shared valid sample subset
- `E22B`: `15-country`, `renewables`, `GRUHybrid`, `window=168`, `h=6`, using the same shared valid sample subset
- `E23`: `20-country`, `public`, `GRUHybrid`, `window=168`, `h=6`
- `E24`: `20-country`, `renewables`, `GRUHybrid`, `window=168`, `h=6`, with missingness-aware window retention

### Why these are the next coded experiments
- `E17A/E17B` make the renewables comparison cleaner than `E15A/E15B`
- `E18` checks whether the renewables gain is stable across random seeds
- `E19` becomes meaningful only after the renewables direction has shown positive evidence
- `E20` attempts to bring the renewables direction back to the full main-country setting without collapsing the sample count
- `E21` does for the hybrid line what `E18` did for the pure-sequence renewables line
- `E22A/E22B` isolate whether hybrid still benefits from renewable features when the architecture is held fixed
- `E23` tests whether hybrid helps the main `20-country + public` benchmark before adding missing-data-heavy renewables
- `E24` is the first full-setup renewables-aware hybrid extension

### Actual outcome from `E17A/E17B`
- This cleaner comparison still favored `renewables`
- The renewables gain is therefore real, though more modest than the original `E15` comparison suggested

### Actual outcome from `E18`
- The renewables lift remained stable across seeds
- The gain is credible but not large enough on its own to change the project's main benchmark hierarchy

### Actual outcome from `E19`
- Hybrid produced the best current result on the matched renewables subset
- This is now the most justified direction for the next round

### Actual outcome from `E20`
- Missingness-aware renewables on `20` countries improved coverage and recall
- It did not beat the main `E12` benchmark on `PR-AUC`
- This branch should not be the immediate focus of the next round

## Decision rules for the next update

### If `E14` improves over `E12`
- Keep `GRU + 168h` as the main deep-learning backbone
- Carry the improved training strategy into richer-feature experiments where appropriate

### Actual outcome from `E14`
- `E14` did not improve the main selection metric `PR-AUC`
- Keep `E12` as the main score benchmark
- Retain `E14` only as a recall-oriented variant

### If `E15B > E15A`
- Conclude that the deep-learning line benefits from renewable features
- Increase confidence in richer-feature sequence experiments

### Actual outcome from `E15A/E15B`
- This condition was satisfied on the current run
- Renewable features are now the most credible next richer-feature direction

### If `E16B > E16A`
- Conclude that the deep-learning line benefits from system-state flow features
- Increase confidence in mechanism-driven feature expansion

### Actual outcome from `E16A/E16B`
- This condition was not satisfied on the current run
- Do not prioritize further flow-feature expansion before fixing the comparison quality

### If richer-feature experiments improve substantially
- Hybrid becomes more justified as a score-maximization and complementarity test

### Actual outcome from `E17-E20`
- The renewables direction is now confirmed
- Hybrid is no longer just a justification question; it is the strongest current next-step line
- Missingness-aware `20-country` renewables remains exploratory rather than decisive

### If richer-feature experiments fail to improve
- Revisit training stability, sequence data pipeline efficiency, and model calibration before expanding architecture complexity

## Maintenance rule
- Whenever a materially new direction judgment is made, update this file so later threads can see the latest recommended roadmap.
