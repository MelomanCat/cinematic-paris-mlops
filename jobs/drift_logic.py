

def rel_change(cur, base):
    if base == 0:
        return 1.0
    return abs(cur - base) / base


def is_city_drifted(cur, base):
    return (
        cur["n_zones"] == 0 or
        rel_change(cur["mean_zone_radius_m"], base["mean_zone_radius_m"]) > 0.30 or
        rel_change(cur["mean_films_per_zone"], base["mean_films_per_zone"]) > 0.30 or
        rel_change(cur["n_zones"], base["n_zones"]) > 0.20
    )