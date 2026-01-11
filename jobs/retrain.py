import pandas as pd
import numpy as np
import boto3
import mlflow
from sqlalchemy import create_engine
from sklearn.cluster import DBSCAN
from zone_metrics import compute_zone_metrics
from datetime import datetime, timezone
import os

# --- Config 
S3_BUCKET = "jedha-lead-bucket"
BASELINE_KEY = "monitoring/reference/baseline_2016_2023.csv"
MODEL_PREFIX = "models/"
TMP_DIR = "/opt/airflow/tmp"

EPS_KM = 0.1
MIN_SAMPLES = 10

MLFLOW_TRACKING_URI = "https://jedha0padavan-mlflow-server-final-project.hf.space"
EXPERIMENT_NAME = "cinematic-paris-hotspots"

DATABASE_URL = os.getenv("INGEST_DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("INGEST_DATABASE_URL environment variable is not set!")

engine = create_engine(DATABASE_URL)
os.makedirs(TMP_DIR, exist_ok=True)

# --- Functions 
def load_baseline():
    s3 = boto3.client("s3")
    local = os.path.join(TMP_DIR, "baseline.csv")
    s3.download_file(S3_BUCKET, BASELINE_KEY, local)
    return pd.read_csv(local)

def load_current_window(limit=2000):
    return pd.read_sql(f"""
        SELECT lat, lon
        FROM filming_events
        ORDER BY year DESC
        LIMIT {limit}
    """, engine)

def train(df):
    coords = np.radians(df[["lat", "lon"]].values)
    eps_rad = EPS_KM / 6371
    model = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric="haversine")
    df["cluster"] = model.fit_predict(coords)
    return model, df

def save_model_to_s3(model, run_id):
    local = os.path.join(TMP_DIR, f"model_{run_id}.pkl")
    pd.to_pickle(model, local)
    s3_key = f"{MODEL_PREFIX}model_{run_id}.pkl"
    boto3.client("s3").upload_file(local, S3_BUCKET, s3_key)
    return s3_key

# --- Main 
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading baseline...")
    baseline = load_baseline()

    print("Loading current events...")
    current = load_current_window()

    base_model, base_df = train(baseline)
    cur_model, cur_df = train(current)

    base_metrics = compute_zone_metrics(base_df)
    cur_metrics = compute_zone_metrics(cur_df)

    # collapse detection
    city_collapsed = cur_metrics["n_zones"] == 0

    # Safe comparison to avoid division by 0
    def rel_change(cur, base):
        if base == 0:
            return 1.0
        return abs(cur - base) / base

    drift = (
        city_collapsed or
        rel_change(cur_metrics["mean_zone_radius_m"], base_metrics["mean_zone_radius_m"]) > 0.30 or
        rel_change(cur_metrics["mean_films_per_zone"], base_metrics["mean_films_per_zone"]) > 0.30 or
        rel_change(cur_metrics["n_zones"], base_metrics["n_zones"]) > 0.20
)

    with mlflow.start_run(run_name="city_evolution_retrain"):
        for k, v in cur_metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_param("eps_km", EPS_KM)
        mlflow.log_param("min_samples", MIN_SAMPLES)
        mlflow.log_param("city_collapsed", city_collapsed)

        if drift:
            print("City evolved → deploying new model")
            key = save_model_to_s3(cur_model, mlflow.active_run().info.run_id)
            mlflow.log_param("model_s3_path", key)
        else:
            print("City stable — no deploy")

if __name__ == "__main__":
    main()