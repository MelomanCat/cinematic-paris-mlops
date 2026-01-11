from jobs.retrain_policy import should_retrain

def test_drift_triggers_retrain():
    base = {
        "n_zones": 10,
        "mean_films_per_zone": 40,
        "mean_zone_radius_m": 300
    }

    current = {
        "n_zones": 10,
        "mean_films_per_zone": 42,
        "mean_zone_radius_m": 500
    }

    assert should_retrain(current, base) is True