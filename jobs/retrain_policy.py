

from jobs.drift_logic import is_city_drifted

def should_retrain(cur, base, force=False):
    if force:
        return True
    if cur is None or base is None:
        return True
    return is_city_drifted(cur, base)