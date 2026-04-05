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

## Maintenance rule
- Whenever a materially new direction judgment is made, update this file so later threads can see the latest recommended roadmap.
