from jobs.drift_logic import is_city_drifted

def test_zone_radius_drift_detected():
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

    assert is_city_drifted(current, base) is True