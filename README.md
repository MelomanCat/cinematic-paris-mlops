📍 Cinematic Paris — End-to-End MLOps Pipeline
🎬 Project Overview

Cinematic Paris is a full end-to-end MLOps system that automatically detects changes in movie shooting patterns in Paris, retrains clustering models when the city evolves, and redeploys a production inference API without manual intervention.

This project demonstrates an ML lifecycle:

Data drift detection

Conditional retraining

Versioned model storage in S3

CI/CD-driven deployment to HuggingFace Spaces

🧠 Architecture
           ┌──────────────┐
           │   Airflow    │
           │ Daily jobs   │
           └──────┬───────┘
                  │
      Zone metrics & drift detection
                  │
        Drift? ───Yes───▶ Retrain model
                  │             │
                  No            ▼
                  │        Save model to S3
                  ▼             │
            Do nothing     Log in MLflow
                                │
                                ▼
                       GitHub CI/CD pipeline
                                │
                                ▼
                      HuggingFace Spaces API

🔁 Automated ML Lifecycle
Stage	                                        Description
Drift detection	                                Checks city evolution based on zone density, radius and volume
Retraining	                                    Model retrains only when real drift is detected
Model storage	                                Versioned pickle models are stored in S3
CI/CD	                                        GitHub Actions builds, tests, smoke-tests and deploys inference API
Inference	                                    FastAPI serves predictions using the latest S3 model


🚀 Inference API
Endpoint
POST /predict

{
  "lat": 48.86,
  "lon": 2.35
}


Response:

{
  "cluster": 3,
  "is_hotspot": true
}


🧪 CI/CD Pipeline

Every push to main triggers:

Run drift & metric tests

Build inference Docker image

Smoke test API

Deploy to HuggingFace Spaces


🗂 Project Structure
cinematic-paris-mlops/
│
├── jobs/
|   |-- drift_logic.py
|   |-- retrain_policy.py
|   |-- retrain.py
|   |--zone_metrics.py
|    
├── inference_api/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── tests/
│   ├── test_drift_logic.py
│   ├── test_retrain_policy.py
│   └── test_zone_metrics.py
│
└── .github/workflows/ci.yml


Project characteristics :

- Fully automated retraining

- Drift-aware deployment

- No manual model promotion

- Production inference served directly from S3


✨ Author

Built by Olga Kosenko
Data Science & MLOps Engineer