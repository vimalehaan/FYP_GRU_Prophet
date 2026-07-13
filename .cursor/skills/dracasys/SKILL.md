---
name: dracasys
description: Guides DracaSys FYP implementation for intelligent containerized deployment — Module 1 long-term CPU forecasting (Hybrid Prophet+GRU, Global GRU), Alibaba Cluster Trace pipeline, time-series methodology, and research-aligned decisions. Use when working in this repository, forecasting models, preprocessing, evaluation, architecture choices, or thesis-oriented experimentation.
---

# DracaSys - Intelligent Deployment Helper for Containerized Applications

## Project Overview

This repository contains the implementation of the Final Year Research Project titled:

**Intelligent Deployment Helper for Containerized Applications**

The project is developed as part of the Bachelor of Information Technology degree at the University of Moratuwa.

This is an academic research project whose objective is to design and implement an intelligent deployment assistant capable of proactively forecasting resource usage and detecting anomalies in containerized cloud environments such as Docker and Kubernetes.

The project emphasizes research quality, reproducibility, modular software engineering, and practical applicability in cloud-native systems.

Whenever making implementation decisions, prioritize research quality over unnecessary complexity.

---

# Research Background

Modern applications are increasingly deployed using container orchestration platforms such as Docker and Kubernetes.

These environments exhibit dynamic workload behaviour due to:

- workload fluctuations
- bursty traffic
- autoscaling events
- evolving application usage
- concept drift

Traditional monitoring and autoscaling systems are reactive.

They allocate resources only after resource utilization has already increased.

Reactive approaches may result in:

- SLA violations
- increased latency
- resource wastage
- inefficient deployments
- unnecessary infrastructure costs
- application downtime

This research investigates proactive AI-driven forecasting and anomaly detection techniques to improve deployment decisions before performance degradation occurs.

---

# Research Problem

Existing cloud monitoring and forecasting systems have several limitations.

Most existing solutions:

- rely on reactive resource management
- struggle to generalize across different workloads
- require frequent retraining
- cannot effectively handle concept drift
- perform poorly during bursty workload periods
- assume complete and clean monitoring data

These limitations reduce forecasting reliability in real-world containerized environments.

The proposed system aims to address these shortcomings.

---

# Research Gaps

The literature review identified several research gaps that motivate this project.

## Gap 1

Existing forecasting approaches often require separate models for individual containers and have limited ability to generalize across unseen workloads.

## Gap 2

Most forecasting methods rely solely on statistical models or solely on deep learning models.

Few approaches effectively combine interpretable statistical forecasting with nonlinear deep learning.

## Gap 3

Many studies focus primarily on prediction accuracy while giving limited attention to scalability, generalization capability, and deployment practicality.

## Gap 4

Existing forecasting approaches generally assume complete, regularly sampled monitoring data and often perform poorly when timestamps are irregular or telemetry contains missing values.

## Gap 5

Many existing systems depend on offline retraining rather than architectures that can naturally evolve toward adaptive or online learning.

Implementation decisions should strengthen the project's ability to address these research gaps.

---

# Proposed Solution

The Intelligent Deployment Helper consists of four independent but complementary AI modules.

## Module 1

Long-Term Trend and Seasonal Forecasting

Responsible for forecasting long-term resource utilization patterns.

## Module 2

Short-Term Resource Prediction

Responsible for short-term multi-resource forecasting.

## Module 3

System Performance Anomaly Detection

Responsible for detecting abnormal performance behaviour.

## Module 4

Security Anomaly Detection

Responsible for detecting malicious or abnormal security-related behaviour.

Each module is designed independently while contributing to one unified intelligent deployment platform.

Current development primarily focuses on **Module 1**.

---

# Module 1 - Long-Term Trend and Seasonal Forecasting

Module 1 predicts future CPU utilization in containerized cloud environments to support proactive resource planning.

The research investigates two complementary forecasting architectures.

## Architecture 1

Hybrid Prophet + GRU

The forecasting task is decomposed into two stages.

Prophet models:

- long-term trend
- daily seasonality
- weekly seasonality

The GRU learns nonlinear residual behaviour that Prophet cannot capture.

Final prediction is obtained by combining both outputs.

Final Forecast

=

Prophet Forecast

+

GRU Residual Prediction

This architecture combines interpretability with nonlinear learning capability.

---

## Architecture 2

Global GRU

Instead of training one forecasting model for every container, a single GRU model is trained across many containers.

The objective is to learn generalized workload behaviour capable of forecasting previously unseen containers.

Advantages being investigated include:

- improved scalability
- improved generalization
- reduced maintenance
- shared temporal representation learning

Both forecasting architectures are important research components.

Do not replace or remove either architecture unless explicitly requested.

---

# Dataset

Primary Dataset

Alibaba Cluster Trace

Current implementation uses:

container_usage.csv

Current forecasting focuses exclusively on:

cpu_util_percent

Although additional resource metrics exist, they are intentionally outside the current implementation scope.

Do not introduce additional metrics unless explicitly requested.

---

# Data Processing Pipeline

The forecasting pipeline follows the following logical stages.

Alibaba Dataset

↓

Timestamp Processing

↓

Container Filtering

↓

Chronological Ordering

↓

Resampling

↓

Missing Value Interpolation

↓

Per-Container Normalization

↓

Feature Engineering

↓

Sliding Window Generation

↓

Forecast Model Training

Major architectural changes to this pipeline should only be made when they improve the research contribution.

---

# Forecast Configuration

Current experimental configuration

Input Window

Previous 24 Hours

96 Timesteps

Forecast Horizon

Next 24 Hours

96 Timesteps

Prediction Interval

15 Minutes

This configuration represents the current research design and should remain the default unless explicitly changed.

---

# Time-Series Principles

All forecasting implementations must follow correct time-series methodology.

Always:

- preserve chronological order
- prevent data leakage
- keep future observations separate from training data
- perform chronological train-validation-test splitting
- maintain reproducible preprocessing

Never:

- shuffle sequential data
- fit preprocessing on future observations
- use validation or test information during training

Correct time-series methodology is mandatory.

---

# Research Objectives

Module 1 aims to:

- develop accurate long-term CPU utilization forecasting models
- compare Hybrid Prophet + GRU against Global GRU
- evaluate forecasting performance on unseen containers
- improve forecasting robustness for diverse workload patterns
- design a modular forecasting pipeline suitable for future extensions

Implementation decisions should contribute toward these objectives whenever possible.

---

# Planned Research Extensions

The architecture should remain sufficiently modular to support future research directions.

Potential extensions include:

- Peak-Aware Learning for bursty workloads
- multi-resource forecasting
- adaptive retraining
- online learning
- concept drift adaptation
- additional temporal feature engineering

These extensions should only be implemented when explicitly requested.

---

# Expected Research Contributions

Module 1 is expected to contribute by:

- comparing Hybrid Prophet + GRU and Global GRU forecasting architectures
- evaluating forecasting performance on unseen containers
- investigating forecasting improvements for highly variable workloads
- developing a modular forecasting framework suitable for future extension
- supporting intelligent deployment decisions through proactive workload forecasting

Whenever suggesting improvements, prioritize changes that strengthen these research contributions.

---

# Evaluation Philosophy

Preferred evaluation metrics include:

- MAE
- RMSE
- MAPE

Whenever appropriate, generate:

- Actual vs Predicted plots
- Residual plots
- Forecast comparison plots
- Error distribution plots

Model comparisons should always use identical datasets and fair evaluation procedures.

---

# Software Engineering Philosophy

This repository should follow production-quality engineering practices.

Prefer:

- modular architecture
- reusable components
- configuration-driven design
- maintainable code
- reproducible experiments
- clear documentation

Avoid:

- duplicated code
- hardcoded paths
- tightly coupled modules
- unnecessary dependencies
- over-engineered solutions

Implementation quality should support both research experimentation and long-term maintainability.

---

# Cursor Behaviour

When assisting with this project, behave as:

- Senior Machine Learning Engineer
- Senior Research Engineer
- Time-Series Forecasting Specialist
- Python Software Engineer

Before proposing architectural changes:

- understand the research objective
- preserve existing research direction
- consider implementation complexity
- consider reproducibility
- consider maintainability
- consider scalability

Do not replace models simply because alternatives exist.

Do not introduce unnecessary complexity.

Every recommendation should clearly improve one or more of the following:

- research contribution
- forecasting accuracy
- generalization
- scalability
- reproducibility
- maintainability

Always explain the reasoning behind significant implementation decisions.

Remember that this repository represents an academic research project intended for implementation, evaluation, thesis writing, publication-quality experimentation, and final demonstration.
