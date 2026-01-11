import pandas as pd
from jobs.zone_metrics import compute_zone_metrics

# test function with no hotspots found
def test_empty_clusters_returns_safe_defaults():
    df = pd.DataFrame({
        "lat": [48.85, 48.86],
        "lon": [2.35, 2.36],
        "cluster": [-1, -1]
    })

    metrics = compute_zone_metrics(df)

    assert metrics["n_zones"] == 0
    assert metrics["mean_films_per_zone"] == 0.0
    assert metrics["mean_zone_radius_m"] == 0.0