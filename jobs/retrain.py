import os
from datetime import datetime, timezone  
import json
import boto3
import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import DBSCAN

from jobs.zone_metrics import compute_zone_metrics
from jobs.retrain_policy import should_retrain

# --- Config
S3_BUCKET = "jedha-lead-bucket"
BASELINE_KEY = "monitoring/reference/baseline_2016_2023.csv"
MODEL_PREFIX = "models/"
ZONES_PREFIX = "models/zones/"
TMP_DIR = os.getenv("TMP_DIR", "/tmp/cinematic-paris")
os.makedirs(TMP_DIR, exist_ok=True)

EPS_KM = 0.1
MIN_SAMPLES = 10

MLFLOW_TRACKING_URI = "https://jedha0padavan-mlflow-server-final-project.hf.space"
EXPERIMENT_NAME = "cinematic-paris-hotspots"

DATABASE_URL = os.getenv("INGEST_DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("INGEST_DATABASE_URL environment variable is not set!")

engine = create_engine(DATABASE_URL)

def load_baseline():
    s3 = boto3.client("s3")
    local = os.path.join(TMP_DIR, "baseline.csv")
    s3.download_file(S3_BUCKET, BASELINE_KEY, local)
    return pd.read_csv(local)


def load_current_window(limit=2000):
    return pd.read_sql(
        f"""
        SELECT lat, lon
        FROM filming_events
        ORDER BY year DESC
        LIMIT {limit}
        """,
        engine
    )


def train(df):
    df = df.copy()
    coords = np.radians(df[["lat", "lon"]].values)
    eps_rad = EPS_KM / 6371
    model = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric="haversine")
    df["cluster"] = model.fit_predict(coords)
    return model, df

EARTH_RADIUS_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two points (degrees)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_M * c

def compute_zone_circles(clustered_df: pd.DataFrame):
    """
    Returns circles per cluster: centroid (lat, lon) + radius_m (max distance to centroid).
    Expects 'lat','lon','cluster' columns. Ignores cluster == -1.
    """
    zones = clustered_df[clustered_df["cluster"] != -1].copy()
    circles = []

    if zones.empty:
        return circles

    for cluster_id, group in zones.groupby("cluster"):
        c_lat = float(group["lat"].mean())
        c_lon = float(group["lon"].mean())

        # max distance from centroid to points in cluster
        dists = haversine_m(group["lat"].values, group["lon"].values, c_lat, c_lon)
        radius_m = float(np.max(dists)) if len(dists) else 0.0

        circles.append({
            "cluster": int(cluster_id),
            "lat": c_lat,
            "lon": c_lon,
            "radius_m": radius_m,
            "n_points": int(len(group)),
        })

    # Optional: stable ordering for diff/debug
    circles.sort(key=lambda x: (x["cluster"]))
    return circles

def save_zones_json_to_s3(circles, run_id, metrics=None):
    payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "eps_km": float(EPS_KM),
        "min_samples": int(MIN_SAMPLES),
        "metrics": metrics or {},
        "zones": circles,
    }

    local = os.path.join(TMP_DIR, f"zones_{run_id}.json")
    with open(local, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    s3_key = f"{ZONES_PREFIX}zones_{run_id}.json"
    boto3.client("s3").upload_file(local, S3_BUCKET, s3_key)
    return s3_key


def save_model_to_s3(model, run_id):
    local = os.path.join(TMP_DIR, f"model_{run_id}.pkl")
    pd.to_pickle(model, local)
    s3_key = f"{MODEL_PREFIX}model_{run_id}.pkl"
    boto3.client("s3").upload_file(local, S3_BUCKET, s3_key)
    return s3_key


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading baseline...")
    baseline = load_baseline()

    print("Loading current events...")
    current = load_current_window()

    # Cluster both windows (for metrics/drift)
    _, base_df = train(baseline)
    _, cur_df = train(current)

    base_metrics = compute_zone_metrics(base_df)
    cur_metrics = compute_zone_metrics(cur_df)

    # Candidate retrain dataset = baseline + current
    full_city = pd.concat([baseline, current], ignore_index=True)
    candidate_model, candidate_df = train(full_city)
    candidate_metrics = compute_zone_metrics(candidate_df)
    
    drift = should_retrain(cur_metrics, base_metrics)

    with mlflow.start_run(run_name="city_evolution_retrain"):
        # log candidate metrics
        for k, v in candidate_metrics.items():
            mlflow.log_metric(k, float(v))

        # log baseline/current for debug
        for k, v in base_metrics.items():
            mlflow.log_metric(f"baseline_{k}", float(v))
        for k, v in cur_metrics.items():
            mlflow.log_metric(f"current_{k}", float(v))

        mlflow.log_param("eps_km", EPS_KM)
        mlflow.log_param("min_samples", MIN_SAMPLES)
        mlflow.log_param("drift", int(bool(drift)))
        mlflow.log_param("baseline_key", BASELINE_KEY)

        if drift:
            print("City evolved → retraining and deploying new model")
            key = save_model_to_s3(candidate_model, mlflow.active_run().info.run_id)
            circles = compute_zone_circles(candidate_df)
            zones_key = save_zones_json_to_s3(circles, mlflow.active_run().info.run_id, metrics=candidate_metrics)
            mlflow.log_param("zones_s3_path", zones_key)
            mlflow.log_param("model_s3_path", key)
        else:
            print("City stable — no deploy")


if __name__ == "__main__":
    main()