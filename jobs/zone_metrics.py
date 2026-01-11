import numpy as np
from sklearn.metrics.pairwise import haversine_distances

EARTH_RADIUS_M = 6371000


def compute_zone_metrics(df):
    """
    Urban stability metrics for Cinematic Paris.
    Used for drift detection & retraining decisions.
    """

    zones = df[df["cluster"] != -1].copy()
    metrics = {}

    if zones.empty:
        # No clusters found — return safe defaults
        return {
            "n_zones": 0,
            "mean_films_per_zone": 0.0,
            "mean_zone_radius_m": 0.0,
        }
    # Number of cinematic zones
    zone_sizes = zones.groupby("cluster").size()
    metrics["n_zones"] = int(zone_sizes.shape[0])

    # Density of zones (how many shootings per hotspot)
    metrics["mean_films_per_zone"] = float(zone_sizes.mean())

    # Walkability radius of zones (meters)
    radii = []

    for _, group in zones.groupby("cluster"):
        coords = np.radians(group[["lat", "lon"]].values)
        center = coords.mean(axis=0).reshape(1, -1)
        dists = haversine_distances(coords, center) * EARTH_RADIUS_M
        radii.append(dists.max()) # max dist to be walked within a single hotspot

    # Force all metrics to be pure python floats (MLflow-safe)
    # Add mean_zone_radius_m BEFORE converting
    metrics["mean_zone_radius_m"] = np.mean(radii) if radii else 0.0

    # Force all metrics to be pure python floats (MLflow-safe)
    for k in metrics:
        if hasattr(metrics[k], "item"):
            metrics[k] = float(metrics[k].item())
        else:
            metrics[k] = float(metrics[k])

    return metrics