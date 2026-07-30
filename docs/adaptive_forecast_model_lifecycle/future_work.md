# Future Work

AFMLF intentionally excludes several directions to keep the research phase isolated.

## In scope for future engineering (not this thesis phase)

- **Production integration** — wire monitoring to live forecast-service logs
- **Alerting** — Slack/PagerDuty on `drift_confirmed` or `needs_investigation`
- **Automated promotion gate** — CI job that requires human approval + evaluation thresholds
- **Prophet caching** — speed up full 99-container walk-forward simulations
- **Labelled drift benchmarks** — synthetic regime injections with known ground-truth drift times

## Explicitly out of scope (different research track)

- **Online learning** — continuous GRU gradient updates at inference time
- **Architecture changes** — new Hybrid variants, Peak-Aware layers, Global GRU comparisons
- **Automatic production overwrite** — violates governance principle

## Adaptive learning contribution

The original proposal's adaptive learning requirement is satisfied by **lifecycle governance**: detect → diagnose → recommend offline retrain → evaluate → human deployment. Future work may add lightweight monitoring adapters without reopening forecasting architecture research.
