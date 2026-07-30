# Evaluation Methodology

## Primary — framework metrics

- Confirmed drift events  
- False alarm rate (defer / no-retrain vs trigger)  
- Detection delay (origins)  
- **Recommendation precision**  
- Deployment success rate  
- Prevented bad deployments  
- Diagnostic class distribution  
- Retraining frequency  
- Decision latency  

## Secondary — forecast metrics (context only)

- Candidate vs incumbent MAE/RMSE on future holdout origins  
- Not headline claims of “AFMLF improves forecasting”  

## Recommendation quality

| Outcome | Label |
|---------|--------|
| Recommend → candidate better | Correct |
| Recommend → candidate worse | Incorrect |
| No recommend → candidate would not help | Correct defer |

See `utils/adaptive_lifecycle/evaluation.py`.
