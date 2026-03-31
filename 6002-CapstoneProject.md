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
