# **System Conditions Behind Negative Electricity Prices in High Renewable European Power Markets:** 

# **A Deep Learning-Based Event Prediction Approach**

# **Background and Motivation**

During the rapid development of public infrastructure and clean energy production facilities, electricity prices in Europe have declined year by year, leading to negative electricity prices, especially after 2020, which have gradually become relatively common. Negative electricity prices indicate that, at certain times, the supply of electricity exceeds the actual market demand. This situation reflects not only the increasing penetration of renewable or clean energy sources but also structural constraints in transmission infrastructure, system flexibility, and cross-border balancing capabilities.

A deeper understanding of the negative electricity price state in the electricity market is beneficial for further improving and designing flexible power systems with high renewable energy penetration, improving the planning and cross-border coordination of circuit transmission, designing more flexible information delivery mechanisms in the electricity market, and better supporting future low-carbon power systems. Previous research and analysis have explored priority effects and price suppression mechanisms in depth, but less attention has been paid to studying the underlying system states that characterize the conditions for negative electricity prices. This project aims to fill this gap.

# **Research Objective**

The primary objective of this research is to identify and learn the system operating conditions under which European electricity markets are likely to experience future negative electricity price events. Rather than forecasting the exact numerical value of electricity prices, this project formulates the task as a binary event prediction problem with explicit temporal ordering.

For each prediction time `t`, the input should contain only information available up to time `t`, such as a rolling historical window of system conditions. The output should refer to a future observation, such as whether a negative price event occurs at time `t + h` or within a predefined future horizon.

* Class 1: Future negative price event  
* Class 0: Future non-negative price outcome

This framing keeps the study focused on both system-state understanding and practical forecasting, while preserving temporal causality and avoiding information leakage.

# **Research Questions**

1. Under what combinations of generation structure, demand levels, cross-border flows, and weather conditions do future negative price events occur?  
2. Can deep learning models automatically learn latent representations of system operating states from multivariate time-series data for future event prediction?  
3. Do high-dimensional temporal representations provide better discrimination of future negative price events than manually engineered features?

# **Data Source and Description**

Our project integrates multiple publicly available datasets for future negative electricity price event prediction.  
**Electricity Price and Generation Data from Kaggle**

## *Electricity price is used to construct the binary target indicating whether a future target hour or future target window experiences negative pricing.*

**Demand and Cross-Border Flow Data from ENTSO-E Transparency Platform**

## *Negative pricing is not solely a function of high renewable output. It reflects system imbalance and transmission constraints. Including demand and flow data allows modeling system-wide operational conditions rather than isolated generation levels.*

**Weather Data from Copernicus Climate Data Store (ERA5)**

## *Weather does not directly determine electricity prices but affects renewable output exogenously.*

**Calendar and Institutional Factors (if needed)**

## *These factors systematically influence demand structure and may amplify negative price risks during low-demand periods.*

## *Implementation note: the ENTSO-E build pipeline uses `python-holidays` to generate country-level legal holiday flags such as `is_holiday_local`.*

**Methodology**

1. Problem Formulation

The dataset will be structured as multivariate time-series samples, where each sample represents a continuous historical window of system operation, such as the previous 24 hours.

The task definition will be fixed before model comparison. For each sample, the input window contains only information observed up to time `t`, while the label refers to a future target at time `t + h` or a clearly defined future time window. A representative label definition is:

`y_(t+h) = 1` if the electricity price at the future target hour is below zero, and `0` otherwise.

If a future time window is used instead of a single target hour, the event rule will be stated explicitly. All data preprocessing, train-validation-test splits, and experiments will preserve temporal causality and avoid leakage from future observations.

2. Deep Learning Methods

In high renewable penetration power systems, electricity prices are determined by complex and context-dependent interactions among multiple operational factors. An increase in solar or wind generation may depress prices under low demand conditions, but the same generation level may not lead to negative prices if demand is high or cross-border transmission capacity is sufficient. Therefore, the relationship between individual variables and price outcomes is not stable across system states. Negative price events emerge from the joint configuration of generation structure, demand intensity, transmission flows, and system flexibility rather than from any single observable factor.

Traditional machine learning methods typically rely on manually engineered features and assume that the system can be adequately described by a fixed set of low-dimensional explanatory variables. This assumption is restrictive in the present context, where the true system state is latent and only indirectly reflected through high-dimensional, time-dependent operational data. Assigning stable weights or pre-defined interaction structures to individual features may fail to capture the nonlinear and evolving dynamics of renewable-dominated electricity markets.

Deep learning is therefore adopted not to assume a single optimal architecture in advance, but to align the modeling framework with the structure of the problem. By learning representations directly from multivariate time-series data, sequence models can extract latent system states from the joint evolution of supply, demand, weather, and cross-border interactions. The general modeling pipeline is a temporal encoder followed by a classifier that outputs the probability of a future negative price event.

3. Baselines and Candidate Models  
* Leakage-free classification baselines will be established before adding model complexity.  
* LSTM / GRU networks can be used as candidate temporal encoders.  
* Temporal Convolutional Networks (TCN) can also be evaluated.  
* Transformer-based time-series encoders may be included if time permits.  
* No single model family will be assumed to be optimal before empirical comparison.

4. Model Evaluation  
* Accuracy  
* Precision / Recall  
* F1-score  
* ROC-AUC  
* Additional attention will be paid to class imbalance and chronological validation performance.

**Expected Contributions**  
This project aims to contribute by:

1. Modeling negative electricity price events as a future event classification task rather than same-time description or price regression.  
2. Demonstrating the value of representation learning in power system economics.  
3. Integrating generation, demand, transmission, weather, and other contextual variables into a unified forecasting framework.  
4. Providing a temporally consistent and leakage-aware setup for future negative price event prediction in 2024–2025 European markets.  
5. Providing empirical insights into operational patterns associated with elevated future negative pricing risk.

**Conclusion**  
Negative electricity prices reflect a deep structural transformation in the European power system amid high renewable energy penetration. This project therefore proposes a deep learning-based event prediction framework that uses only past and current system information to predict future negative price events. By integrating market, operational, meteorological, and institutional data, the study aims to learn the system states associated with elevated future negative-price risk while maintaining a clear and leakage-free temporal design.

# **Next Experiment Plan**

The next experiment phase is designed to strengthen the deep learning track under the fixed forecasting setup. The main benchmark setting remains future negative-price event prediction with `h = 6`, leakage-free chronological splits, and inputs restricted to information available up to time `t`.

The current implementation uses a rolling historical input window of `72` hours. In the plan below, a longer window means a longer history visible to the model, such as `120` or `168` past hours before the forecast anchor time.

## **Phase 1: Identify the best deep learning backbone**

The first step is to determine whether a longer historical window improves deep sequence modeling under the main `20-market + public features + h = 6` setting.

1. `E11`: `GRU`, `window = 120`, `h = 6`, main 20-market public-feature setting.  
   Goal: test whether extending the input window beyond `72` hours improves event prediction.
2. `E12`: `GRU`, `window = 168`, `h = 6`, main 20-market public-feature setting.  
   Goal: test whether one full week of historical context is more informative than shorter windows.
3. `E13`: `TCN`, `window = 168`, `h = 6`, main 20-market public-feature setting.  
   Goal: compare a convolutional temporal encoder against `GRU` under the longer-window setup.

These three experiments are intended to answer two questions:

* Does a longer input window improve deep learning performance for future negative-price event prediction?  
* Which deep temporal encoder is the strongest candidate for follow-up experiments?

## **Phase 2: Build on the best result from E11-E13**

Experiments `E14-E16` should not be fixed to one architecture in advance. They should inherit the best-performing deep learning backbone and window length identified by `E11-E13`, using validation and test evidence under the same causal setup.

4. `E14`: best model from `E11-E13` plus class-imbalance-focused training refinement.  
   Suggested first option: focal loss or another imbalance-aware training strategy while keeping the same task definition.  
   Goal: improve detection of relatively rare negative-price events and raise `PR-AUC`, `recall`, and `F1`.
5. `E15`: best model from `E11-E13` applied to the renewables-enhanced feature setting with `h = 6`.  
   Goal: test whether a stronger deep sequence model benefits more from richer renewable-related temporal signals.
6. `E16`: best model from `E11-E13` applied to the cross-border-flow feature setting with `h = 6`.  
   Goal: test whether the selected deep model can better exploit system-state information when transmission-flow variables are included.

## **Evaluation logic for E11-E16**

The evaluation protocol remains unchanged:

* preserve target-time-based chronological splitting;
* preserve leakage-free preprocessing and sample construction;
* use future event classification rather than price regression;
* compare models primarily by `PR-AUC`, with `F1`, `recall`, `ROC-AUC`, and balanced accuracy as supporting metrics;
* inspect overall, monthly, and country-level stability rather than relying on one aggregate score only.

The decision rule for continuation is:

* `E11-E13` select the best deep model and best window length;
* `E14-E16` extend that selected model rather than introducing unnecessary architectural changes at the same time.

This staged design keeps the experiment story coherent: first identify whether longer historical context helps deep learning, then improve imbalance handling, and finally test whether the selected deep encoder gains more from richer system-condition feature groups.

## **Phase 3: Cleaner renewables validation and scale-up**

Based on the completed `E14-E16` results, the next deep-learning phase should focus on the renewables direction rather than expanding the flow track immediately.

7. `E17A`: `15-country`, `public`, `GRU`, `window = 168`, `h = 6`, but filtered to the renewables-shared valid-sample subset.  
   Goal: build a stricter public baseline on exactly the same valid sample pool used for the renewables comparison.
8. `E17B`: `15-country`, `renewables`, `GRU`, `window = 168`, `h = 6`, using the same renewables-shared valid-sample subset.  
   Goal: test whether renewable features still improve the deep-learning line under a cleaner matched-sample comparison.
9. `E18`: repeated-seed version of `E17B`.  
   Goal: assess whether the observed gain from the renewables direction is stable across random seeds rather than a single-run fluctuation.
10. `E19`: `15-country`, `renewables`, `GRUHybrid`, `window = 168`, `h = 6`.  
    Goal: test whether a hybrid model combining learned sequence representations and handcrafted tabular summaries can improve over pure sequence modeling.
11. `E20`: `20-country`, `renewables`, `GRU`, `window = 168`, `h = 6`, with missingness-aware window retention.  
    Goal: bring the renewables-enhanced deep-learning line back to the full main-country setting without dropping a large fraction of samples because of incomplete renewable coverage.

# **Experiment Record**

## **Completed deep-learning follow-up experiments**

The `E11-E16` sequence experiments have now been run under the leakage-free future event prediction setup.

### **Stage 1: Longer-window backbone selection (`E11-E13`)**

- `E11`: `GRU`, `window = 120`, `20-market`, `public`, `h = 6`
- `E12`: `GRU`, `window = 168`, `20-market`, `public`, `h = 6`
- `E13`: `TCN`, `window = 168`, `20-market`, `public`, `h = 6`

The key finding from this stage is that `GRU + 168h` (`E12`) became the strongest deep-learning configuration under the main `20-market + public + h = 6` setting. This supports the view that a longer history window is useful for the current forecasting task, and it establishes `E12` as the deep-learning reference line for later follow-up experiments.

### **Stage 2: Imbalance-aware and richer-feature follow-up (`E14-E16`)**

- `E14`: `20-country`, `public`, `GRU`, `window = 168`, `h = 6`, `focal loss`
- `E15A`: `15-country`, `public`, `GRU`, `window = 168`, `h = 6`
- `E15B`: `15-country`, `renewables`, `GRU`, `window = 168`, `h = 6`
- `E16A`: `7-country`, `public`, `GRU`, `window = 168`, `h = 6`
- `E16B`: `7-country`, `flows`, `GRU`, `window = 168`, `h = 6`

The main test-set results are:

| Experiment | Setting | PR-AUC | F1 | Recall | Balanced Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `E12` | `20-country`, `public`, `GRU`, `168h` | `0.2918` | `0.3273` | `0.5911` | `0.7767` |
| `E14` | `20-country`, `public`, `GRU`, `168h`, `focal loss` | `0.2844` | `0.3285` | `0.7076` | `0.8295` |
| `E15A` | `15-country`, `public`, `GRU`, `168h` | `0.2219` | `0.2990` | `0.4739` | `0.7204` |
| `E15B` | `15-country`, `renewables`, `GRU`, `168h` | `0.2665` | `0.3186` | `0.6394` | `0.7963` |
| `E16A` | `7-country`, `public`, `GRU`, `168h` | `0.5192` | `0.5489` | `0.6083` | `0.7981` |
| `E16B` | `7-country`, `flows`, `GRU`, `168h` | `0.3848` | `0.4568` | `0.6459` | `0.8103` |

## **Interpretation of the completed results**

### **1. `E14` improves recall but not the primary metric**

Compared with `E12`, `E14` slightly reduces test `PR-AUC` from `0.2918` to `0.2844`, while test `recall` increases from `0.5911` to `0.7076` and balanced accuracy also improves. This suggests that focal loss makes the model more aggressive in identifying future negative-price events, but it does not improve ranking quality under the main project metric. Therefore, `E14` is better interpreted as a recall-oriented operating variant rather than a replacement for the `E12` baseline.

### **2. Renewable features provide the clearest positive signal**

Within the matched `15-country` comparison, `E15B` outperforms `E15A` on test `PR-AUC`, `F1`, `recall`, and balanced accuracy. This is the strongest evidence so far that a deep sequence model can gain predictive value from renewable-related temporal features. The renewables direction is therefore the most promising richer-feature extension for the current deep-learning line.

### **3. The current flow-feature extension is not yet successful**

Within the matched `7-country` comparison, `E16B` does not beat `E16A`. Although `E16B` achieves slightly higher recall and balanced accuracy, it is materially worse on test `PR-AUC`, precision, and `F1`. This means that the current flow-feature formulation does not yet provide a stronger deep-learning setup than the corresponding public-feature baseline on the same country subset.

### **4. Absolute scores across country subsets should not be over-interpreted**

`E16A` delivers the strongest absolute score in this round, but it is a `7-country` subset experiment rather than the full `20-country` main setup. In addition, the richer-feature runs use fewer valid samples because of feature availability. For example, `E15B` contains fewer samples than `E15A`, and `E16B` contains fewer samples than `E16A`. Therefore, the `E15/E16` results should be interpreted as direction signals rather than perfectly clean causal comparisons.

## **Updated experimental judgment**

Based on the completed `E11-E16` results, the current evidence supports the following conclusions:

1. `E12` should remain the main deep-learning reference under the full `20-country + public + h = 6` setup.
2. `E14` is useful when higher recall is desired, but it does not replace `E12` under a `PR-AUC`-first selection rule.
3. The renewables track is the most promising next direction for improving deep-learning performance.
4. The current flow-feature track should be deprioritized unless missing-data handling or comparison quality is improved.
5. A hybrid model that combines sequence representations with handcrafted features should be considered only after the richer-feature direction has been clarified more cleanly.
