📍 Cinematic Paris — End-to-End MLOps Pipeline
🎬 Project Overview

Cinematic Paris is a full end-to-end MLOps system that automatically detects changes in movie shooting patterns in Paris, retrains clustering models when the city evolves, and redeploys a production inference API without manual intervention.

This project demonstrates an ML lifecycle:

Data drift detection

Conditional retraining

Versioned model storage in S3

CI/CD-driven deployment to HuggingFace Spaces


```text
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
```

🔁 Automated ML Lifecycle
Stage	                                        Description
Drift detection	                                Checks city evolution based on zone density, radius and volume
Retraining	                                    Model retrains only when real drift is detected
Model storage	                                Versioned pickle models are stored in S3
CI/CD	                                        GitHub Actions builds, tests, smoke-tests and deploys inference API
Inference	                                    FastAPI serves predictions using the latest S3 model


🚀 Inference API

The inference API uses the latest zone artifacts (circles: centroid + radius) stored in S3 to:

- Serve the latest cinematic zones for visualization

- Check whether a given location belongs to a cinematic hotspot

Zone artifacts are automatically produced during retraining and versioned in S3.

Endpoints
GET /zones

Returns the latest cinematic zones as circles.

Response example:

{
  "run_id": "b3a91f...",
  "created_at_utc": "2026-01-12T10:43:21Z",
  "metrics": {
    "n_zones": 12,
    "mean_films_per_zone": 38.5,
    "mean_zone_radius_m": 210.3
  },
  "zones": [
    {
      "cluster": 0,
      "lat": 48.861,
      "lon": 2.349,
      "radius_m": 180.4,
      "n_points": 42
    },
    {
      "cluster": 1,
      "lat": 48.853,
      "lon": 2.372,
      "radius_m": 230.1,
      "n_points": 35
    }
  ]
}

This endpoint is used by the visualization layer to draw cinematic zones on a map.

---------------------------------------------------------------------------------

POST /predict

Checks whether a given location is inside a cinematic hotspot.

Request:
{
  "lat": 48.86,
  "lon": 2.35
}

Response example:

{
  "is_hotspot": true,
  "nearest_cluster": 0,
  "distance_m": 42.7,
  "zone_radius_m": 180.4,
  "run_id": "b3a91f..."
}

Logic:

The API finds the nearest cinematic zone (centroid).

Computes the haversine distance to that zone.

If the distance is smaller than the zone radius → is_hotspot = true.

Summary:

The inference API always uses the latest zone artifact from S3.

Zones are versioned and produced automatically during retraining.

Visualization is based on GET /zones, not on model prediction.

POST /predict is a helper endpoint for point-based hotspot checks.



🧪 CI/CD Pipeline

Every push to main triggers:

Run drift & metric tests

Build inference Docker image

Smoke test API

Deploy to HuggingFace Spaces


🗂 Project Structure
```text
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
```


Project characteristics :

- Fully automated retraining

- Drift-aware deployment

- No manual model promotion

- Production inference served directly from S3


✨ Author

Built by Olga Kosenko
Data Science & MLOps Engineer