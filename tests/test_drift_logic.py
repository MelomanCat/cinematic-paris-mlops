from jobs.drift_logic import is_city_drifted

BASE = {
    "n_zones": 10,
    "mean_films_per_zone": 40.0,
    "mean_zone_radius_m": 300.0
}

def test_zone_radius_drift_detected():
    current = {**BASE, "mean_zone_radius_m": 500.0}  # +66%
    assert is_city_drifted(current, BASE) is True

def test_films_per_zone_drift_detected():
    current = {**BASE, "mean_films_per_zone": 60.0}  # +50%
    assert is_city_drifted(current, BASE) is True

def test_n_zones_drift_detected():
    current = {**BASE, "n_zones": 13}  # +30% (>20%)
    assert is_city_drifted(current, BASE) is True

def test_city_collapsed_detected():
    current = {**BASE, "n_zones": 0}
    assert is_city_drifted(current, BASE) is True

def test_no_drift_returns_false():
    current = {**BASE, "mean_zone_radius_m": 350.0, "mean_films_per_zone": 45.0, "n_zones": 11}
    # radius +16%, films +12.5%, zones +10% -> below thresholds
    assert is_city_drifted(current, BASE) is False