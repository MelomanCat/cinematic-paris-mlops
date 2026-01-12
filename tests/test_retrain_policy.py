from jobs.retrain_policy import should_retrain

BASE = {
    "n_zones": 10,
    "mean_films_per_zone": 40.0,
    "mean_zone_radius_m": 300.0
}

def test_drift_triggers_retrain():
    current = {**BASE, "mean_zone_radius_m": 500.0}
    assert should_retrain(current, BASE) is True

def test_force_triggers_retrain_even_without_drift():
    current = {**BASE}
    assert should_retrain(current, BASE, force=True) is True

def test_none_current_triggers_retrain():
    assert should_retrain(None, BASE) is True

def test_none_base_triggers_retrain():
    assert should_retrain(BASE, None) is True

def test_no_drift_returns_false():
    current = {**BASE, "mean_zone_radius_m": 320.0, "mean_films_per_zone": 44.0, "n_zones": 10}
    assert should_retrain(current, BASE) is False