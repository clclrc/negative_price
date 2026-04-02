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
  implemented as a config-backed experiment definition and ready to run

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
- Do not prioritize this immediately
- Add hybrid experiments only after `E14-E16` clarify whether the `E12` line improves under better loss design and richer feature groups

## Recommended order

1. `E14`
2. `E15A`
3. `E15B`
4. `E16A`
5. `E16B`
6. Reassess whether a hybrid experiment should be introduced next

## Decision rules for the next update

### If `E14` improves over `E12`
- Keep `GRU + 168h` as the main deep-learning backbone
- Carry the improved training strategy into richer-feature experiments where appropriate

### If `E15B > E15A`
- Conclude that the deep-learning line benefits from renewable features
- Increase confidence in richer-feature sequence experiments

### If `E16B > E16A`
- Conclude that the deep-learning line benefits from system-state flow features
- Increase confidence in mechanism-driven feature expansion

### If richer-feature experiments improve substantially
- Hybrid becomes more justified as a score-maximization and complementarity test

### If richer-feature experiments fail to improve
- Revisit training stability, sequence data pipeline efficiency, and model calibration before expanding architecture complexity

## Maintenance rule
- Whenever a materially new direction judgment is made, update this file so later threads can see the latest recommended roadmap.
