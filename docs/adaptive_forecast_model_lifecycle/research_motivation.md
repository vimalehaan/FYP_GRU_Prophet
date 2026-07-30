# Research Motivation

Forecasting research for Module 1 is **complete**. The Hybrid Prophet+GRU model was selected after controlled comparisons (Global GRU, Peak-Aware variants, diagnostic programmes).

The original proposal identified **adaptive learning and concept drift management** as a contribution. AFMLF satisfies that contribution as **lifecycle governance**, not as a new forecaster.

## Problem

After deployment:

- **Prophet** partially adapts (refit on latest history each request)  
- **GRU weights** and **global residual statistics** remain frozen  
- Workload regimes evolve → forecast errors may degrade  

Operators need a principled answer to: *Should we retrain offline? Should we deploy a new artifact version?*

## AFMLF answer

Monitor rolling forecast errors → detect drift per container → confirm statistically → diagnose Prophet vs Hybrid → trigger cohort policy → train candidate offline → evaluate on future data → recommend deployment (human gate).
