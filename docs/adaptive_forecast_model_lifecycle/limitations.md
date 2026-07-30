# Limitations

- Historical walk-forward simulation (Alibaba trace), not live telemetry  
- Prophet fit per origin is computationally expensive  
- Global GRU retrain is coarse when only subset of containers drift  
- CSRLE suggests limited GRU headroom — framework may be **correct** while MAE gains are small  
- Page-Hinkley sensitivity on short error streams  
- Human required for actual deployment  

See [future_work.md](future_work.md) for extension directions.
