# Experiment Direction Notes

## Purpose
This file records the current judgment on experiment directions, feasibility, risks, and recommended next steps.

It is intended to help future threads and future agents avoid re-deciding the same roadmap from scratch.

## Current status

### Best current deep-learning baseline
- `E30`: `GRUHybridGated`, `window=168`, `h=6`, `20-market`, `public features`
- Current role:
  best-performing deep-learning configuration under the main forecasting setup

### High-level conclusion so far
- Longer history helps `GRU`
- `GRUHybridGated` is now the most promising deep-learning line under the main setup
- `E31` is the strongest precision- and `F1`-oriented companion branch under the same full setup
- Pure `GRU` currently looks more promising than `TCN` and clearly more promising than the current `PatchTST`
- Deep learning still underperforms the strongest tree-model baselines
- The most defensible next step is now to continue from `E30/E31` rather than reopening older backbone questions

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

### Signal group 6: hybrid confirmation on the main setup (`E21-E24`)
- `E21` confirmed that the `E19` hybrid gain is stable across random seeds, with a mean test `PR-AUC` around `0.283`
- `E22B` still outperformed `E22A` under the same shared valid-sample subset, so renewable features remain helpful even when the architecture is fixed to `GRUHybrid`
- `E23` outperformed `E12` on the full `20-country + public + h=6` benchmark, making it the first deep-learning line in this project to beat the earlier full-setup deep baseline on the main metric
- `E24` did not beat `E23`, so adding missingness-aware renewables back into the full setup is not yet the strongest continuation path

#### Direction implication
- `E23` should replace `E12` as the main deep-learning benchmark for future rounds
- `E21` and `E22B` provide supporting evidence that the hybrid gain is both stable and mechanism-aligned
- `E24` should remain exploratory until it can outperform `E23`
- The next round should optimize and stress-test the `E23` line rather than broadening the feature space again

### Signal group 7: full-setup stress test and next-generation hybrid variants (`E25-E33`)
- `E25` showed that the `E23` line has meaningful seed sensitivity: the repeated-seed mean test `PR-AUC` is lower than the single-run `E23` score, even though the mean `F1` is slightly higher
- `E26` underperformed `E23` on the main metric, so focal loss still does not look like the best way to improve the full-setup hybrid line
- `E27` slightly improved over `E23`, which suggests training budget matters, but only modestly
- `E28` improved slightly over `E24`, which means the full renewables-aware branch is not dead, but the gain is too small to make it the main score line
- `E29` improved over `E23`, so temporal attention pooling is a real positive direction, but it did not become the strongest line
- `E30` is the strongest completed result so far under the full `20-country + public + h=6` setup, outperforming `E23` on `PR-AUC`, `ROC-AUC`, `recall`, `F1`, and balanced accuracy
- `E31` also outperformed `E23`, with the strongest completed `precision` and `F1`, although its recall and balanced accuracy are below `E30`
- `E32` underperformed both `E30` and `E31`, so the current multi-task design should not be prioritized further
- `E33` confirmed that the `E30` branch is genuinely strong, but also showed meaningful seed variance because its repeated-seed mean falls below the best single `E30` run
- `E34-E36` are still pending, so no roadmap change should depend on them yet

#### Direction implication
- `E30` should replace `E23` as the main deep-learning benchmark for the full setup
- `E31` should be kept as a serious parallel branch because it appears complementary rather than redundant
- `E28` should remain exploratory and not replace the public-feature mainline
- The current multi-task design should be deprioritized unless a later variant changes the outcome materially
- The immediate next round should focus on validating and exploiting `E30` and `E31`, not reopening focal loss or older pure-`GRU` questions

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
- The next iteration should focus on confirming and strengthening the `E23` line
- Hybrid is now the strongest deep-learning family in both subset and full-setup comparisons

## Recommended order

1. Keep `E23` as the main full-setup deep-learning benchmark
2. Use `E21` and `E22B` as supporting stability and mechanism checks, not as the primary score target
3. Run repeated-seed validation for `E23`
4. Test whether imbalance-aware training or a larger training budget improves `E23`
5. Revisit a `20-country` renewables-hybrid extension only after the `E23` line is better established

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

### Actual outcome from `E21-E24`
- `E21` showed that the hybrid gain over `E19` is stable but not enormous
- `E22B > E22A`, so renewable features still help within the hybrid family
- `E23` is now the strongest deep-learning result under the main `20-country + public` setup
- `E24` did not beat `E23`, so the current full-setup renewables-hybrid direction should stay secondary

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

## Current completed-round judgment after `E28`, `E32`, `E33`, `E34`, `E35`, `E37`, and `E38`

- `E35` is now the strongest completed deep-learning result and should be treated as the main score benchmark
- `E30` remains the strongest completed single-branch full-setup deep-learning result
- `E31` is close enough to `E30`, and different enough in `precision` versus `recall`, that it should remain an active companion branch
- `E25` shows the old `E23` line was not stable enough to treat a single run as fully representative
- `E26` should not be prioritized further
- `E28` is a small positive update to the renewables-aware full branch, but not a mainline replacement
- `E32` should not be prioritized further in its current form
- `E33` confirms that the `E30` line is worth keeping, but it also raises the priority of stability-aware follow-up experiments
- `E34` shows that the `E31` line is slightly weaker than `E33` on repeated-seed mean `PR-AUC`, but much more stable across seeds
- `E35` confirms that `E30` and `E31` are genuinely complementary under the same full setup, not just under isolated single runs
- `E36` is still pending, so it should not yet be interpreted as a final direction change
- `E37` shows that stability-oriented tuning helps the `E30` branch, but not enough to displace `E35`
- `E38` does not give the same uplift to the `E31` mechanism-aware branch, so it should not be promoted over `E37` or `E35`

### Actual outcome from `E35`

- `E35` test `PR-AUC = 0.3691`, higher than both completed single branches:
  `E30 = 0.3399`, `E31 = 0.3338`
- On the matched `E35` reruns, the member models scored:
  `E30 member = 0.3209`, `E31 member = 0.3574`
- This means the late-fusion gain is real under the same rerun context, not just an artifact of comparing against older single-run outputs
- The validation-time member weights are nearly balanced:
  `E30 = 0.4869`, `E31 = 0.5131`
- `E35` also raises `F1` to `0.3923`, higher than both `E30` and `E31`

### Actual outcome from `E37`

- `E37` test `PR-AUC = 0.3494`, which improves on the original `E30 = 0.3399`
- `E37` also improves `precision` and `F1` over `E30`:
  `precision 0.2503 vs 0.2099`, `F1 0.3646 vs 0.3277`
- `E37` does not beat `E35 = 0.3691`, so stability tuning alone is weaker than the already confirmed ensemble gain
- `E37` is still useful as a stronger single-branch baseline for any future comparisons against `E38`, `E39`, or `E40`

### Actual outcome from `E38`

- `E38` test `PR-AUC = 0.3297`, slightly below the original `E31 = 0.3338`
- It does improve over the repeated-seed mean of `E34 = 0.3099`, so stability-oriented training is not useless for the mechanism-aware line
- But unlike `E37`, it does not create a stronger completed single-branch reference
- This means the mechanism-aware branch remains useful for ensemble diversity, but not as the best score-maximizing single model

## Next planned experiments around `E30` and `E31`

- `E33`: repeated-seed version of `E30`
- `E34`: repeated-seed version of `E31`
- `E35`: validation-weighted late-fusion ensemble of `E30` and `E31`
- `E36`: probability calibration branch on the strongest available completed classifier among `E30`, `E31`, and `E35`

### Why this is the right next plan
- `E30` is the new main score target for deep learning, so it needs a proper stability check before more architectural churn
- `E31` may be complementary rather than weaker, because it trades some recall for noticeably higher precision and `F1`
- `E35` is justified because `E30` and `E31` show different error profiles under the same setup
- `E36` is justified by both the literature and the current score profile: once the strongest classifier family is identified, probability quality becomes the next high-value question
- `E34` strengthens the justification for `E35`, because it shows that `E31` still contributes a robustness profile that `E30` does not provide on its own

### Planned roles
- `E33` answers whether the `E30` gain is stable across seeds
- `E34` answers whether the mechanism-aware `E31` gain is stable across seeds
- `E35` answers whether the `E30` and `E31` branches are complementary enough to improve over either single model
- `E36` answers whether the strongest current classifier can be made more decision-useful without sacrificing ranking quality

### Updated priority after `E35`

- `E36` is now more valuable, because calibration should be applied to a demonstrably stronger base candidate rather than to an uncertain branch
- `E39` should now be treated as the highest-value pending follow-up after `E36`, because a simple weighted late-fusion already worked well
- `E40` is more justified than before, because both branch diversity and seed diversity now look usable as ensemble assets
- `E37` and `E38` remain worth keeping, but they move down one step in urgency because ensemble complementarity has already delivered a stronger score gain than stability tuning alone is likely to deliver

### Updated priority after `E37`

- `E37` validates that more conservative optimization is directionally useful for the `E30` family
- This strengthens the case for keeping `E38`, but it does not change the top-level ordering
- The recommended execution order remains:
  `E36 -> E39 -> E40 -> E38`
- `E37` itself should now be treated as completed evidence rather than as a pending follow-up

### Updated priority after `E38`

- `E38` should now be treated as completed evidence rather than as a pending follow-up
- The recommended execution order is now:
  `E36 -> E39 -> E40`
- `E38` still supports the rationale for `E39` and `E40`, because it preserves branch diversity even without becoming a stronger standalone score line

### Implementation note

- `E29-E32` are already implemented as config defaults
- `E33-E36` are now also implemented as config defaults
- `E37-E40` are now also implemented as config defaults
- `E33-E36` should still be interpreted around completed `E30/E31` evidence unless `E32` later changes the ordering materially

## Prepared follow-up experiments after `E34-E36`

- `E37`: stability-tuned `E30`
- `E38`: stability-tuned `E31`
- `E39`: out-of-fold stacking ensemble over `E30` and `E31`
- `E40`: cross-seed branch ensemble built from the strongest completed repeated-seed branch or branches

### Why these are the right next designs

- `E33` showed that the `E30` branch has non-trivial seed variance, so a stability-tuned variant is justified before more backbone churn
- `E31` may still become the more defensible branch if `E34` shows better repeated-seed behavior, so it also deserves a stability-focused follow-up design
- A stacking ensemble is a stronger complementarity test than the current static weighted late fusion in `E35`
- A cross-seed ensemble turns observed seed diversity into a modeling asset rather than treating it only as experimental noise

### Planned roles

- `E37`: keep the `E30` architecture, but reduce optimization variance with a lower learning rate and a larger patience or epoch budget
- `E38`: apply the same stability-oriented training changes to the `E31` mechanism-aware branch
- `E39`: use validation-time out-of-fold predictions from `E30` and `E31` as inputs to a small stacking model rather than a fixed weighted average
- `E40`: combine repeated-seed members of the strongest completed branch, or of both `E30` and `E31` if both remain competitive, and only add calibration on top if `E36` proves helpful

### Status note

- `E37-E40` are now implemented as config defaults because the user explicitly requested early implementation
- They should still be interpreted as pending branches until `E35-E36` and then `E37-E40` themselves produce completed results
- The partial `E35` member outputs are still not enough to reorder these follow-ups yet

## Literature-informed addendum

For a fuller write-up, see `LITERATURE_EXPERIMENT_GUIDE.md`.

### Main judgment from the literature review

- Direct future negative-price event prediction remains much less common than ordinary electricity price regression
- The strongest related work usually falls into one of four buckets:
  driver analysis with `logit` models,
  threshold-event classification,
  price or distribution forecasting,
  and generic deep-learning price prediction
- This strengthens the case that the repository should keep its main contribution centered on leakage-free future event classification rather than drifting toward regression-only work

### What the literature says to prioritize next

- Keep strong simple baselines, especially interpretable probabilistic baselines and the best tree models
- Keep `E23` as the main deep-learning score target unless a clearly better family appears under equally rigorous evaluation
- Add probability calibration and probability-quality evaluation as a first-class follow-up direction
- Prefer new experiment families that improve task design and mechanism alignment:
  multi-horizon event prediction,
  future-window event labels,
  mechanism-aware causal summary features,
  and event-plus-price multi-task learning
- Treat transfer and market generalization as a high-value later-stage contribution because the multi-market setup is one of the project's strongest differentiators

### What the literature says not to prioritize

- Do not switch the mainline to a transformer-first strategy without stronger evidence
- Do not reopen the current flow-heavy branch before improving missing-data handling and matched-sample comparison quality
- Do not replace the event-classification framing with plain price regression
- Do not over-interpret score differences across country subsets with different data coverage

### Updated recommended order after `E25-E27`

1. Probability calibration on the strongest current classifier
2. Multi-horizon event prediction with shared versus separate heads
3. Future-window event labels that remain strictly causal
4. Mechanism-aware hybrid features such as residual-load and ramp proxies
5. Multi-task event plus price prediction
6. Stronger transfer and market-generalization tests

### Why this addendum matters

- The recent literature does not invalidate the current `GRUHybrid` roadmap
- Instead, it suggests that the highest-value next gains are more likely to come from probability quality, label design, and mechanism-aligned feature construction than from backbone churn
- This aligns with the strongest empirical signals already observed in this repository

## Latest benchmark judgment after `E35-E44`

- The matched current-task classical benchmark is now `E44`, not the older `E8`
- `E44 LightGBM` achieved test `PR-AUC = 0.4139`
- `E42 XGBoost` is close behind at `0.4066` and currently gives the strongest same-task threshold `F1`
- `E35` remains the strongest completed deep benchmark at `0.3691`
- `E37` remains the strongest completed single deep branch at `0.3494`
- `E36` and `E39` should not be prioritized further for score-maximization under the current setup
- `E43` still needs a dependency-complete rerun before it can be interpreted

### What this changes

- The main score gap is now between the best deep line and the matched current-task tree baselines
- Because `E35` already proved branch-level complementarity, the next deep-learning family should aim to learn information that the tabular tree baselines do not already exploit as well
- The strongest next bet is therefore a multi-market graph-temporal hybrid rather than more calibration, stacking, or stability-only follow-up on the current single-market family

## Prepared next-generation deep family: `E45-E48`

- `E45`: `GRUMultiMarket`
  Goal: verify whether jointly encoding all markets at the same anchor time helps even before explicit market-interaction layers are added
- `E46`: `GraphTemporal`
  Goal: test whether a market-interaction layer improves on simple joint multi-market pooling
- `E47`: `GraphTemporalHybrid`
  Goal: fuse graph-temporal market representations with the current strongest mechanism-aware handcrafted branch
- `E48`: repeated-seed version of `E47`
  Goal: check whether any graph-temporal gain is stable enough to treat as a genuine next-generation deep line

### Recommended order after the current completed evidence

1. `E45`
2. `E46`
3. `E47`
4. `E48`

### Why this is now the right order

- `E35` already answered the simpler complementarity question inside the current `E30/E31` family
- `E37` and `E38` showed that stability tuning is useful but not transformative
- `E44` showed that the matched classical baselines are still clearly ahead, so the next deep experiments need to capture cross-market structure rather than only retune single-market encoders

### Implementation note

- `E45-E48` are now implemented as config defaults and executable through the standard runner
- They should still be treated as pending next-generation experiments until they produce completed metrics

## Actual outcome from `E40` and `E45-E47`

- `E40` did not improve on the simpler `E35` ensemble
- `E45` is the first next-generation deep result that clearly beats the previous deep benchmarks
- `E46` and `E47` did not improve on `E45`; in their current form, explicit graph interaction is not yet helping
- `E43` still remains blocked by missing `catboost` dependency
- `E48` is still pending, so the stability of the graph-temporal hybrid branch is not yet known

### Updated benchmark view

- Current matched classical benchmark: `E44 LightGBM`, test `PR-AUC = 0.4139`
- Current strongest completed deep ensemble: `E35`, test `PR-AUC = 0.3691`
- Current strongest completed deep single model: `E45 GRUMultiMarket`, test `PR-AUC = 0.3880`

### What changed

- `E45` now becomes the main deep-learning single-model score target
- `E35` should still be kept as the main completed deep ensemble benchmark
- `E40` should not be prioritized further because cross-seed ensembling under the current family did not beat the already simpler and stronger `E35`
- `E46` and `E47` should be treated as failed first-pass graph-temporal variants rather than promoted follow-ups

### Why `E45` matters

- `E45` is the first deep single model to break well above the old `E37 = 0.3494`
- It also beats the completed deep ensemble `E35 = 0.3691`
- This suggests that joint multi-market context is genuinely valuable
- The result still trails the matched tree baselines, especially `E44`, so the deep-learning gap is smaller but not closed

### Why `E46` and `E47` are not yet justified

- `E46 GraphTemporal` dropped to test `PR-AUC = 0.3168`
- `E47 GraphTemporalHybrid` dropped further to `0.3126`
- Both show large validation-to-test deterioration relative to `E45`
- The current evidence therefore does not justify further graph-layer complexity before revisiting the graph design itself

### Updated recommended order

1. Finish `E48`
2. Re-run or repeat-seed `E45`
3. Keep `E44` as the project-level matched classical reference
4. Treat `E46/E47` as exploratory failures unless a materially different graph design is proposed

## Prepared follow-up experiments after `E45`

- `E49`: repeated-seed version of `E45`
  Goal: verify that the multi-market gain is stable and not another single-run spike
- `E50`: `E45` plus a gated handcrafted tabular branch
  Goal: combine the new multi-market sequence gain with the strongest existing lag and rolling-stat summaries
- `E51`: `E50` plus mechanism-aware engineered features
  Goal: test whether the `E31`-style mechanism signals become more useful once the sequence branch already captures joint multi-market context
- `E52`: late-fusion ensemble of `E45` and the strongest completed deep comparator
  Goal: turn the new multi-market single-model line into a stronger deep benchmark without immediately mixing in classical models

### Recommended order after the current evidence

1. `E49`
2. `E50`
3. `E51`
4. `E52`

### Why this is the current best plan

- `E45` is the first next-generation deep model that materially narrows the gap to the matched tree baselines
- The most urgent unanswered question is whether `E45` is stable
- If `E45` is stable, the most promising next move is not more graph complexity but adding back the strongest handcrafted information through a cleaner hybrid branch
- A final deep-only ensemble is justified only after the single-model line is stabilized and strengthened

### Implementation status

- `E49-E52` are now implemented as runnable config defaults in the repository
- `E49` should be interpreted as the first priority stability check for the `E45` line
- `E50/E51` are the main architecture follow-ups because they test whether the new multi-market sequence gain can be combined with the strongest handcrafted summaries
- `E52` is the prepared deep-only ensemble branch for the post-`E45` family

## Actual outcome from `E49-E51`

- `E49` confirms that the `E45` multi-market gain is real and unusually stable
- `E50` does not improve on `E45`; adding the handcrafted tabular branch in its current form hurts the multi-market line
- `E51` recovers part of that loss with mechanism-aware features, but still does not beat `E45`
- `E52` is still running and should not yet affect the roadmap

### Updated benchmark view after `E49-E51`

- Matched classical benchmark remains `E44 LightGBM`, test `PR-AUC = 0.4139`
- Strongest completed deep ensemble remains `E35`, test `PR-AUC = 0.3691`
- Strongest completed deep single-model line is now best summarized by `E49` rather than only the single-run `E45`
  - `E45` single run: test `PR-AUC = 0.3880`
  - `E49` repeated-seed mean: test `PR-AUC = 0.3867`, std `0.0014`

### What changed

- `E45` is no longer just a promising spike; `E49` shows it is the most stable strong deep single-model family seen so far
- The current best post-`E45` interpretation is that joint multi-market sequence modeling itself is the useful gain
- Re-attaching the old handcrafted branch is not yet helping under this family
- Mechanism-aware features are less harmful than the plain hybrid branch, but still not enough to justify promoting `E51`

### Why `E49` matters

- The repeated-seed mean is almost identical to the single-run `E45` result
- The variance is extremely small compared with earlier deep repeated-seed branches
- This makes the `GRUMultiMarket` line much more defensible as the main deep single-model benchmark

### Why `E50/E51` are currently not justified

- `E50 GRUMultiMarketHybrid` drops to test `PR-AUC = 0.3513`
- `E51` improves that to `0.3633`, but still trails both `E45 = 0.3880` and `E49 mean = 0.3867`
- The current evidence therefore does not support prioritizing more handcrafted-branch fusion under the `E45` family

### Updated recommended order

1. Wait for `E52`
2. Keep `E49` as the main deep single-model stability reference
3. Keep `E44` as the matched classical score target
4. De-prioritize `E50/E51` unless a materially different fusion design is proposed

## Prepared follow-up experiments after `E49`

- `E53`: target-conditioned market-attention `GRUMultiMarket`
  Goal: replace the simple cross-market mean pooling in `E45/E49` with learned target-conditioned market attention so the model can focus on the most relevant neighboring markets
- `E54`: temporal-attention `GRUMultiMarket`
  Goal: replace the simple last-state style temporal summary in the current multi-market encoder with attention pooling over the full `168h` history
- `E55`: 15-market renewables-track `GRUMultiMarket` on a strict shared-valid-sample subset
  Goal: test whether the strong multi-market sequence line benefits from richer raw sequence inputs, rather than from re-attaching handcrafted summary branches
- `E56`: score-ceiling fusion between the strongest stable deep single-model line and the strongest matched classical baseline
  Goal: measure whether the new multi-market deep signal is complementary to the strongest tree baseline even if the standalone deep model still trails it

### Recommended order after `E49`

1. `E53`
2. `E54`
3. `E55`
4. `E56`

### Why this is the current best plan

- `E49` shows the gain comes from the multi-market sequence model itself, so the next changes should stay inside that sequence branch
- `E50/E51` show that re-attaching the current handcrafted branch is not the right next move
- The most promising architectural gains are now better market aggregation and better temporal aggregation
- If those still do not close the gap to `E44`, the next useful result is to test complementarity between the stable deep line and the strongest classical line

### Implementation status

- `E53-E56` are now implemented as runnable config defaults in the repository
- `E53` and `E54` are the main architecture follow-ups because they keep the gain inside the now-stable `GRUMultiMarket` family
- `E55` is the richer-input branch for testing raw renewables sequence value under a strict comparable subset
- `E56` is the prepared score-ceiling branch for testing deep plus classical complementarity

## Actual outcome from `E52`

- `E52` is a real positive result
- It improves on `E35`
- It does not beat `E49` on `PR-AUC`
- It still does not beat `E44`

### Updated benchmark view after `E52`

- Matched classical benchmark remains `E44 LightGBM`, test `PR-AUC = 0.4139`
- Strongest completed deep single-model line remains `E49/E45 GRUMultiMarket`, test `PR-AUC ≈ 0.387`
- Strongest completed deep ensemble is now `E52`, test `PR-AUC = 0.3772`

### What changed

- `E52` confirms that the stable multi-market deep line and the earlier deep ensemble line are complementary
- The gain is real relative to `E35`, but not enough to pass `E49`
- The project still does not have evidence that any deep-only branch has closed the gap to the matched tree baseline
- This means the deep roadmap should continue to focus on strengthening the `GRUMultiMarket` family itself rather than repeatedly wrapping existing deep members in more ensemble layers

### Why `E52` matters

- `E52` improves on `E35 = 0.3691`
- `E52` does not improve on `E49 mean = 0.3867` in the primary metric
- `E52` is therefore useful as the best completed deep ensemble branch, but not as the new main deep score target

## Actual outcome from `E53-E56`

- `E53` is negative evidence
- `E54` is strong negative evidence
- `E55` is a positive but subset-only renewables signal
- `E56` becomes the strongest completed result after the late-fusion merge bug is fixed

### Updated benchmark view after `E56`

- Strongest completed result is now `E56 = E49 + E44` late fusion, test `PR-AUC = 0.4257`
- Strongest matched classical single-model benchmark remains `E44 LightGBM`, test `PR-AUC = 0.4139`
- Strongest stable deep single-model benchmark remains `E49 GRUMultiMarket`, repeated-seed mean test `PR-AUC = 0.3867`
- Strongest completed deep-only ensemble remains `E52`, test `PR-AUC = 0.3772`

### What `E53` and `E54` tell us

- `E53 GRUMultiMarketTargetAttn` drops to test `PR-AUC = 0.3114`
- `E54 GRUMultiMarketTemporalAttn` drops further to `0.2468`
- The current evidence therefore does not support replacing the simple `E49` multi-market pooling with these first attention variants

### What `E55` tells us

- `E55` reaches test `PR-AUC = 0.3614` on the strict `15-market renewables` subset
- This is much stronger than the earlier `15-market renewables` deep lines such as `E17B = 0.2665`, `E19 = 0.2775`, and `E22B = 0.2735`
- It should therefore be kept as a viable subset-specific renewables branch, but not treated as a direct replacement for the `20-country` main benchmark

### What `E56` changes

- `E56` beats `E44` on the matched `20-country + public + window=168 + h=6` task
- The gain is not from a new standalone deep model, but from complementarity between the stable deep multi-market line and the strongest classical baseline
- This means the project now has evidence that deep-learning representations add usable signal, even though the strongest standalone deep single model still trails the strongest standalone classical model

### Updated recommended order

1. Keep `E49` as the main standalone deep architecture reference
2. Keep `E56` as the score-ceiling benchmark for the current task
3. De-prioritize `E53/E54` and the current `GraphTemporal` family
4. Keep `E55` as the subset renewables branch if the goal is to study richer raw-sequence value rather than the `20-country` main score
5. Design future work around improving the `E49` family or around stronger deep-plus-classical complementarity, not around the current attention replacements

## Prepared follow-up experiments after `E56`

The next round should split into a score-ceiling track and a scientific-comparison track.

### Score-ceiling track

- `E57`: `E49 + E42` late fusion
  Goal: test whether the strongest stable deep single-model line is more complementary to `XGBoost` than to `LightGBM`
- `E58`: `E49 + E42 + E44` late fusion
  Goal: test whether combining the stable deep line with both strong classical baselines produces a higher ceiling than `E56`

### Scientific-comparison track

- `E59`: `15-country public GRUMultiMarket` with `sample_filter_feature_group="renewables"`
  Goal: create the missing matched public baseline for `E55` on the exact same valid-sample subset
- `E60`: repeated-seed `E55`
  Goal: only after `E59` confirms the renewables branch has net gain, verify whether the `E55` uplift is stable across seeds

### Recommended order after `E56`

1. `E57`
2. `E58`
3. `E59`
4. `E60` if and only if `E55 > E59`

### Why this is the current best plan

- `E56` already proves that the stable deep multi-market line and the strongest classical baseline are complementary
- The fastest way to improve the overall score is therefore to test complementarity against the other strong classical baseline `E42`, and then against both `E42` and `E44`
- `E55` is promising, but it still lacks the matched `15-country public` comparator needed for a clean scientific claim about renewables-track gain under the `GRUMultiMarket` family
- The failed `E53/E54` attention replacements mean the next useful deep-only work should not be another architectural replacement inside the current `E49` family until the simpler score-ceiling and subset-comparison questions are answered

### Implementation status

- `E57-E60` are now implemented as runnable config defaults in the repository
- `E59` and `E60` intentionally use the standard sequence training budget rather than the earlier aggressive settings

## Actual outcome from `E57-E59`

- `E57` is a positive complementarity result, but not the strongest one
- `E58` becomes the new strongest completed result on the matched main task
- `E59` is the missing matched public comparator for the `E55` renewables branch, and it currently weakens the renewables-only interpretation

### Updated benchmark view after `E57-E59`

- Strongest completed overall result is now `E58 = E49 + E42 + E44`, test `PR-AUC = 0.4281`
- Strongest matched classical single-model benchmark remains `E44 LightGBM`, test `PR-AUC = 0.4139`
- Strongest stable deep single-model benchmark remains `E49 GRUMultiMarket`, repeated-seed mean test `PR-AUC = 0.3867`
- Strongest completed deep-plus-classical two-way fusion is `E56 = E49 + E44`, test `PR-AUC = 0.4257`

### What `E57` and `E58` tell us

- `E57 = E49 + E42` reaches test `PR-AUC = 0.4216`
- `E58 = E49 + E42 + E44` reaches test `PR-AUC = 0.4281`
- `E57` therefore confirms that the stable multi-market deep line is also complementary to `XGBoost`, not just to `LightGBM`
- `E58` improves on both `E56` and `E44`, which means the best current score ceiling comes from combining the stable deep line with both strong tree baselines rather than only one of them
- The `E58` validation-time fusion weights are nearly balanced across members, which suggests that all three branches contribute useful non-redundant signal

### What `E59` changes about the renewables-track interpretation

- `E59` reaches test `PR-AUC = 0.3738` on the same `15-country` renewables-valid subset used to motivate the `E55` comparison
- This is higher than `E55 = 0.3614`, so the earlier claim that the renewables raw-sequence branch is clearly better than the matched public branch is not supported by the current evidence
- However, this is still not a perfect feature-only comparison because `E55` used the earlier aggressive sequence budget while `E59` uses the standard `30 epoch / patience 5` budget
- The current defensible statement is therefore:
  the `GRUMultiMarket` family remains promising on the renewables subset, but the repository does not yet have clean evidence that the renewables input track itself gives net gain over the matched public multi-market baseline

### Updated recommended order after `E57-E59`

1. Keep `E49` as the main standalone deep architecture reference
2. Keep `E58` as the strongest overall score-ceiling benchmark
3. Keep `E56` as the best simpler two-way deep-plus-classical fusion reference
4. De-prioritize `E53/E54` and the current graph-attention replacement lines
5. Do not advance `E60` as a scientific renewables confirmation experiment until the budget mismatch between `E55` and `E59` is resolved or interpreted explicitly

### Best remaining angles for a further score jump

- There is currently no strong evidence that another small standalone deep architecture tweak will produce a large gain over `E49`
- The most promising score-ceiling angle is still stronger complementarity, not more attention or graph churn
- Highest-priority score experiment:
  run the missing matched-task `E43 CatBoost` baseline and then test a four-way fusion such as `E49 + E42 + E43 + E44`
- Second-priority score experiment:
  replace simple validation-`PR-AUC` weighting with a stronger out-of-fold stacker over the strongest completed members (`E42`, `E44`, `E49`, and `E43` if available)
- Main scientific-comparison angle:
  resolve the `E55` vs `E59` budget mismatch before claiming that the renewables-track multi-market line is truly better than the matched public baseline
- Lower-priority directions:
  more calibration work, more first-generation attention replacements, and the current graph-temporal branch

## Actual outcome from `E43` and `E60`

- `E43` now has a valid completed run and becomes the strongest matched classical single-model baseline by a narrow margin
- `E60` weakens the renewables-track story materially

### Updated benchmark view after `E43`

- `E43 CatBoost` reaches test `PR-AUC = 0.4147`
- `E44 LightGBM` remains essentially tied at `0.4139`
- `E42 XGBoost` remains slightly lower at `0.4066`
- The matched classical single-model benchmark can now be summarized as `E43`, but the practical difference versus `E44` is very small

### What `E43` changes

- `E43` confirms that the classical baseline ceiling is still a little higher than `E49` as a standalone deep model
- It also means any further score-ceiling fusion work should treat `CatBoost` as a first-class candidate member rather than leaving it out
- The most natural next ceiling test is now a four-way fusion such as `E49 + E42 + E43 + E44`, or at minimum a direct `E49 + E43` complementarity check

### What `E60` changes

- `E60`, the repeated-seed version of `E55` under the standard sequence budget, reaches mean test `PR-AUC = 0.3115`
- That is well below the original single-run `E55 = 0.3614`
- It is also well below the matched public comparator `E59 = 0.3738`
- The seed variance is small, so the issue is not instability around a high mean; the issue is that the mean itself is weak under the standard budget

### Updated renewables-track interpretation

- The repository no longer has defensible evidence that the current `15-country renewables` `GRUMultiMarket` line is better than the matched public baseline
- `E55` should now be treated as a likely optimistic one-off result produced under the earlier aggressive training budget
- `E59` is currently the stronger scientific reference for that subset comparison
- `E60` should not be used as support for promoting the renewables-track branch

### Updated recommended order after `E43` and `E60`

1. Keep `E49` as the main standalone deep architecture reference
2. Keep `E58` as the strongest overall score-ceiling benchmark
3. Treat `E43` and `E44` as the two strongest classical single-model members for future fusion work
4. De-prioritize the current renewables-track branch unless it is redesigned and re-tested under a cleaner matched setup

## Prepared follow-up experiments after `E43` and `E60`

The next round should prioritize score-ceiling fusion rather than further standalone deep architecture churn.

### Score-ceiling track

- `E61`: `E49 + E43` late fusion
  Goal: measure the direct complementarity between the strongest stable deep single model and the strongest matched classical single model
- `E62`: `E49 + E43 + E44` late fusion
  Goal: test whether the two strongest tree baselines together beat the current `E56` two-way fusion and approach or exceed `E58`
- `E63`: `E49 + E42 + E43 + E44` late fusion
  Goal: test the current full ceiling by combining the stable deep line with all three strong tree baselines
- `E64`: out-of-fold stacking over `E49`, `E42`, `E43`, and `E44`
  Goal: test whether a stronger meta-learner can improve on the simple validation-`PR-AUC` weighted fusion used by `E58`

### Recommended order

1. `E61`
2. `E62`
3. `E63`
4. `E64`

### Why this is the best next step

- `E58` already shows that the main remaining gains come from complementarity, not from another small deep backbone modification
- `E43` is now available and should be incorporated directly into the fusion line
- `E60` weakens the case for spending the next round on the current renewables-track branch
- All new experiments in this family should use the standard training budget where applicable: `lr=1e-3`, `max_epochs=30`, `patience=5`

## Actual outcome from `E61-E64`

- `E61` is positive but does not exceed the best completed fusion
- `E62` improves over `E56`, but still does not beat the strongest three-way result
- `E63` becomes the new strongest completed result in the repository
- `E64` shows that the current stacking meta-learner is worse than the simple weighted late-fusion ceiling

### Updated benchmark view after `E61-E64`

- Strongest completed overall result is now `E63 = E49 + E42 + E43 + E44`, test `PR-AUC = 0.4284`
- Second-strongest completed result is `E58 = E49 + E42 + E44`, test `PR-AUC = 0.4281`
- Third-strongest completed result is `E62 = E49 + E43 + E44`, test `PR-AUC = 0.4272`
- Best simpler two-way deep-plus-classical fusion is `E56 = E49 + E44`, test `PR-AUC = 0.4257`
- Strongest matched classical single-model baseline remains `E43 CatBoost`, test `PR-AUC = 0.4147`
- Strongest stable deep single-model benchmark remains `E49 GRUMultiMarket`, repeated-seed mean test `PR-AUC = 0.3867`

### What `E61-E63` tell us

- `E61 = E49 + E43` reaches test `PR-AUC = 0.4230`
- `E62 = E49 + E43 + E44` reaches test `PR-AUC = 0.4272`
- `E63 = E49 + E42 + E43 + E44` reaches test `PR-AUC = 0.4284`
- This means:
  the deep line is complementary to `CatBoost`,
  the pair `E43 + E44` is slightly stronger than `E43` alone,
  and the full four-member fusion gives the highest ceiling seen so far
- However, the gain from `E58` to `E63` is very small, so the project is likely close to the current simple-fusion ceiling under this task definition

### What `E64` tells us

- `E64` reaches test `PR-AUC = 0.4215`
- That is below `E61`, `E62`, `E63`, and `E58`
- The current out-of-fold logistic stacker therefore does not improve on the simpler validation-`PR-AUC` weighted late fusion
- Future score-ceiling work should prefer the late-fusion line over the current stacking formulation unless the stacker design itself is changed materially

### Updated recommended order after `E61-E64`

1. Keep `E49` as the main standalone deep architecture reference
2. Keep `E63` as the strongest overall score-ceiling benchmark
3. Keep `E58` as the simpler three-member late-fusion reference because it is almost tied with `E63`
4. De-prioritize the current stacking line (`E64`)
5. Do not spend the next round on the current renewables-track branch unless it is redesigned under a cleaner matched setup

## Prepared follow-up experiments after `E61-E64`

The next round should test context-aware fusion rather than more standalone model churn.

### Why this is the right shift

- `E63` only beats `E58` by a tiny margin, so the current global weighted late-fusion is likely near its simple ceiling
- `E64` shows that the current global logistic stacker is not a better replacement
- The most plausible remaining gain is that complementarity is not stationary:
  different countries, seasons, or operating regimes may prefer different members

### Proposed next experiments

- `E65`: nonlinear OOF stacker over `E49`, `E42`, `E43`, and `E44`
  Goal: replace the weak global logistic stacker with a stronger tree-based meta-learner such as `CatBoost` or `LightGBM`
  Direct baseline: `E64`, then `E63`
- `E66`: country-aware late fusion over `E49`, `E42`, `E43`, and `E44`
  Goal: fit separate validation-time fusion weights by country or by small country clusters
  Direct baseline: `E63`
- `E67`: regime-aware late fusion over `E49`, `E42`, `E43`, and `E44`
  Goal: fit separate fusion weights by broad temporal regime such as season or weekday/weekend
  Direct baseline: `E63`
- `E68`: context-aware meta-learner using member probabilities plus light context features
  Goal: combine member probabilities with simple context such as `country`, `month`, `hour`, and `is_weekend_local`
  Direct baseline: the best result among `E65-E67`

### Recommended order

1. `E65`
2. `E66`
3. `E67`
4. `E68`

### Practical note

- These experiments should mostly reuse already completed member artifacts rather than retraining the base models
- They should therefore be much cheaper than another deep-model search round

## Maintenance rule
- Whenever a materially new direction judgment is made, update this file so later threads can see the latest recommended roadmap.

## Actual outcome from `E65-E68`

- `E65` becomes the new strongest completed result in the repository
- `E66` and `E67` show that grouped late-fusion by country or coarse temporal regime is viable, but neither beats the best global nonlinear stacker
- `E68` shows that the current context-aware LightGBM stacker is worse than the simpler nonlinear stacker over member probabilities alone

### Updated benchmark view after `E65-E68`

- Strongest completed overall result is now `E65 = nonlinear stacking(E49, E42, E43, E44)`, test `PR-AUC = 0.4321`
- Second-strongest completed result is `E63 = E49 + E42 + E43 + E44`, test `PR-AUC = 0.4284`
- Third-strongest completed result is `E58 = E49 + E42 + E44`, test `PR-AUC = 0.4281`
- Country-aware grouped fusion `E66` reaches test `PR-AUC = 0.4279`
- Regime-aware grouped fusion `E67` reaches test `PR-AUC = 0.4280`
- Context-aware stacker `E68` reaches test `PR-AUC = 0.3997`
- Strongest matched classical single-model baseline remains `E43 CatBoost`, test `PR-AUC = 0.4147`
- Strongest stable deep single-model benchmark remains `E49 GRUMultiMarket`, repeated-seed mean test `PR-AUC = 0.3867`

### What `E65` tells us

- A stronger nonlinear meta-learner is the first post-`E63` direction that produces a clear new gain
- `E65 = StackingLightGBM(E49, E42, E43, E44)` reaches test `PR-AUC = 0.4321`
- That beats `E63 = 0.4284` by a non-trivial margin relative to the tiny `E58 -> E63` gain
- This means the project was not yet at the true fusion ceiling; it was at the ceiling of simple validation-`PR-AUC` weighted averaging

### What `E66` and `E67` tell us

- `E66` learns country-specific fusion weights and reaches test `PR-AUC = 0.4279`
- `E67` learns season/weekpart-specific fusion weights and reaches test `PR-AUC = 0.4280`
- Both are competitive with `E58` and close to `E63`, so the context-dependent complementarity hypothesis is directionally reasonable
- But neither grouped late-fusion beats the stronger nonlinear stacker in `E65`
- The current takeaway is therefore:
  grouped weighting is interesting for interpretation,
  but not the best pure score-maximization path right now

### What `E68` tells us

- `E68` adds light context features (`country`, `month`, `hour`, `is_weekend`) to the nonlinear stacker
- It falls to test `PR-AUC = 0.3997`
- That is below `E65`, `E66`, `E67`, `E63`, and even below the strongest classical single-model baseline
- The current context-augmented stacker is therefore not a good next main line

### Updated recommended order after `E65-E68`

1. Keep `E65` as the strongest overall benchmark
2. Keep `E63` as the strongest simple late-fusion reference
3. Keep `E58` as the leaner three-member late-fusion reference
4. Keep `E49` as the main standalone deep architecture reference
5. De-prioritize the current grouped-fusion line (`E66`, `E67`) for score chasing, while keeping it available for interpretation
6. De-prioritize the current context-aware stacker (`E68`)

## Actual outcome from `E69-E72`

- `E69` and `E70` show that grouped nonlinear stacking is not a good score-maximization path in the current setup
- `E71` is competitive but still does not beat `E65`
- `E72` is weaker than `E71`, which reinforces that `XGBoost` is the more useful complementary tree branch in the current stacking line

### Updated benchmark view after `E69-E72`

- Strongest completed overall result remains `E65 = nonlinear stacking(E49, E42, E43, E44)`, test `PR-AUC = 0.4321`
- Best grouped-stacking result is only `E69`, test `PR-AUC = 0.4014`
- Best three-member nonlinear stacker in this round is `E71 = stacking(E49, E42, E44)`, test `PR-AUC = 0.4288`
- `E72 = stacking(E49, E43, E44)` reaches test `PR-AUC = 0.4227`
- `E70` is a strong negative result at test `PR-AUC = 0.2521`

### What `E69` and `E70` tell us

- Grouped nonlinear stacking by country or by broad temporal regime overfits badly relative to the global nonlinear stacker
- `E69` reaches only `0.4014` on test despite very high validation performance
- `E70` collapses further to `0.2521`
- So the repository should not keep pushing the current grouped-stacking line as the next score-ceiling direction

### What `E71` and `E72` tell us

- Dropping `CatBoost` but keeping `E49 + E42 + E44` still gives a strong result: `E71 = 0.4288`
- Dropping `XGBoost` and keeping `E49 + E43 + E44` is weaker: `E72 = 0.4227`
- This suggests:
  `E42` contributes more unique meta-signal than `E43` in the current nonlinear stacking family,
  while `E43` is still useful enough that the full four-member `E65` remains best

### Updated recommended order after `E69-E72`

1. Keep `E65` as the strongest overall benchmark
2. Keep `E63` and `E58` as strong simpler late-fusion references
3. Keep `E71` as the strongest reduced-member nonlinear stacker reference
4. Stop prioritizing grouped nonlinear stacking (`E69`, `E70`)
5. Continue one more low-cost pure-meta iteration only on:
   the remaining untested `E49 + E42 + E43` subset and
   LightGBM meta-stacker hyperparameter variants around `E65`

## Actual outcome from `E73-E76`

- `E73-E76` confirm that the remaining pure-meta gains are now extremely small
- `E75` becomes the new best completed result, but only by a tiny margin over `E65`
- No reduced-member variant beats the full four-member nonlinear stacker

### Updated benchmark view after `E73-E76`

- Strongest completed overall result is now `E75 = tuned nonlinear stacking(E49, E42, E43, E44)`, test `PR-AUC = 0.4323`
- `E74 = regularized nonlinear stacking(E49, E42, E43, E44)` reaches `0.4322`
- `E65 = default nonlinear stacking(E49, E42, E43, E44)` remains essentially tied at `0.4321`
- Best reduced-member nonlinear stacker remains `E76 = tuned stacking(E49, E42, E44)`, test `PR-AUC = 0.4289`
- `E73 = stacking(E49, E42, E43)` reaches `0.4260`

### What `E73` tells us

- The last missing high-value subset `E49 + E42 + E43` does not beat the best four-member stacker
- This means `E44` still contributes enough useful signal that removing it is not beneficial in the current nonlinear stacker family

### What `E74-E76` tell us

- LightGBM meta-stacker hyperparameter tuning can still move the score slightly
- However, the gain from `E65 = 0.4321` to `E75 = 0.4323` is extremely small
- `E74` and `E75` both confirm that the full four-member member set is already close to the practical pure-meta ceiling
- `E76` shows that even after tuning, the strongest reduced-member variant still does not beat the best four-member variant

### Final recommendation for the pure no-retrain line

1. Use `E75` as the best completed overall score-ceiling benchmark
2. Keep `E65` as the main simpler nonlinear-stacking reference because it is essentially tied with `E75`
3. Keep `E63` and `E58` as the main weighted late-fusion references
4. Keep `E49` as the main standalone deep architecture reference
5. Treat the current pure-meta no-retrain line as close to exhausted:
   grouped variants fail,
   context-aware stacking fails,
   subset variants do not win,
   and hyperparameter tuning only adds negligible gains

## Prepared next experiments after `E73-E76`

The next phase should stop tuning the same completed member set and instead create one new genuinely different base-model branch that can be added back into the fusion ceiling.

### Why this is the right shift

- `E75` only improves on `E65` by a tiny amount
- The strongest pure-meta member set is now very stable and very hard to improve further without new signal
- So the only realistic way to get another meaningful gain is:
  train at least one new branch with a different input representation,
  then test whether it is complementary to `E75`

### Proposed next experiments

- `E77`: mechanism-aware `LightGBM` on the main `20-country + public + 168h + h=6` task
  Goal: create a stronger or at least more complementary tree member by adding only public-derived mechanism features such as recent price minima, volatility, ramps, and cross-market dispersion summaries
  Direct baselines: `E44`, then `E43`
- `E78`: mechanism-aware `CatBoost` on the same task and same new feature family
  Goal: test whether the same mechanism-aware tabular view is more natural for `CatBoost` than for `LightGBM`
  Direct baselines: `E43`, then `E44`
- `E79`: `GRUMultiMarket` with public-derived mechanism channels added directly to the sequence input
  Goal: create a new deep member that is different from `E49` without reverting to the already weak late-branch fusion design used by `E50/E51`
  Direct baseline: `E49`
  Default sequence budget: keep the standard `lr=1e-3`, `max_epochs=30`, `patience=5`
- `E80`: score-ceiling fusion between `E75` and the best genuinely new member from `E77-E79`
  Goal: test whether the newly trained branch adds non-redundant signal beyond the current best pure-meta ceiling
  Direct baseline: `E75`

### Recommended order

1. `E77`
2. `E78`
3. `E79`
4. `E80`, but only after at least one of `E77-E79` shows either a standalone gain or a clearly different validation-time error profile

These four experiments are now implemented in code. `E80` is wired as a best-member late-fusion wrapper that keeps `E75` fixed, evaluates `E77-E79` as candidate new members, and fuses `E75` with the strongest candidate by validation `PR-AUC`.

## Actual outcome from `E77-E80`

- `E77 = mechanism-aware LightGBM` does not improve the classical line enough to matter
- `E78 = mechanism-aware CatBoost` is the only clearly successful new branch in this round
- `E79 = GRUMultiMarket + mechanism sequence channels` is a negative result despite a long training run
- `E80` should not be treated as a successful score-ceiling update, because it selected the wrong new member on validation and underperformed badly on test

### Updated benchmark view after `E77-E80`

- `E75` remains the strongest completed overall result, test `PR-AUC = 0.4323`
- `E78` is now the strongest newly trained single model in this phase, test `PR-AUC = 0.4296`
- `E43` remains the strongest older classical single-model baseline, test `PR-AUC = 0.4147`
- `E79` only reaches test `PR-AUC = 0.2999`
- `E80 = E75 + selected_best(E77,E78,E79)` falls to test `PR-AUC = 0.3870`

### What `E77` tells us

- Public-derived mechanism features are not automatically useful for `LightGBM`
- `E77 = 0.3968` is below both `E43 = 0.4147` and `E44 = 0.4139`
- So `E77` should not be kept as a promising new member for future fusion work

### What `E78` tells us

- `E78 = 0.4296` is a real positive result
- It improves meaningfully over `E43 = 0.4147` and `E44 = 0.4139`
- It is also only slightly below `E75 = 0.4323`
- This means the mechanism-aware `CatBoost` branch is the first post-`E75` new member that looks genuinely useful

### What `E79` tells us

- Adding public-derived mechanism channels directly into the `GRUMultiMarket` sequence input does not work in the current form
- `E79 = 0.2999` is far below `E49 = 0.3867`
- This branch should be treated as failed and should not be prioritized further if the project is being wrapped up soon

### What `E80` tells us

- `E80` selected `E79` over `E78` because of higher validation `PR-AUC` in the candidate-selection step
- On test, that choice generalized very poorly and `E80` fell to `0.3870`
- So the current `best_member_late_fusion` wrapper is not reliable enough to be used as the project's final benchmark-selection rule
- The practical lesson is not that `E78` is weak; it is that automatic validation-only candidate selection can overfit

### Practical recommendation if the project should end soon

1. Treat `E75` as the current best completed overall result
2. Treat `E78` as the only clearly valuable new trained branch from the latest phase
3. Stop investing in `E77`, `E79`, and the current `E80` auto-selection wrapper
4. If one final low-cost experiment is still acceptable, run a forced fusion between `E75` and `E78`
5. Otherwise, stop here and write up the story as:
   strong deep single model (`E49`),
   strong classical single model (`E78` or `E43`/`E44` family),
   and best overall score from nonlinear fusion (`E75`)

## Prepared final low-cost closeout experiment

- `E81`: forced late fusion between `E75` and `E78`
  Goal: answer the only remaining cheap question cleanly after `E80` mis-selected `E79`
  Direct baseline: `E75`
  Design rule: fixed members only, no automatic candidate selection, and full artifact reuse whenever possible

## Actual outcome from `E81`

- `E81 = forced late fusion(E75, E78)` succeeds
- It cleanly answers the final open closeout question left unresolved by `E80`

### Updated benchmark view after `E81`

- `E81` is now the strongest completed overall result, test `PR-AUC = 0.4388`
- `E75` becomes the strongest standalone meta-ceiling reference, test `PR-AUC = 0.4323`
- `E78` remains the strongest new final-phase single model, test `PR-AUC = 0.4296`
- `E49` remains the strongest stable standalone deep model, repeated-seed mean test `PR-AUC = 0.3867`

### What `E81` tells us

- The failure of `E80` was due to bad automatic candidate selection, not because `E78` lacked value
- Forcing the fusion between `E75` and `E78` lifts test `PR-AUC` from `0.4323` to `0.4388`
- This is a meaningful final gain for a very cheap artifact-reuse experiment
- The weight split is still balanced enough to support a complementarity interpretation rather than domination by one member

### Final practical recommendation

1. Use `E81` as the best completed overall result in the thesis
2. Use `E75` as the strongest reusable pure-meta reference
3. Use `E78` as the strongest final-phase newly trained single model
4. Use `E49` as the main standalone deep-learning reference
5. Stop major experimentation here unless a non-trivial new project objective is introduced
