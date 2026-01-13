from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import boto3, os, json
import numpy as np

S3_BUCKET = os.getenv("S3_BUCKET", "jedha-lead-bucket")
ZONES_PREFIX = os.getenv("ZONES_PREFIX", "models/zones/")
TMP_DIR = os.getenv("TMP_DIR", "/tmp/cinematic-paris")

EARTH_RADIUS_M = 6371000.0

app = FastAPI(title="Cinematic Paris Hotspot API")

class Location(BaseModel):
    lat: float
    lon: float

_zones_payload = None

def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(EARTH_RADIUS_M * c)

def load_latest_zones():
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=ZONES_PREFIX)

    if "Contents" not in resp:
        raise RuntimeError("No zones found in S3. Cannot start zones-based inference.")

    # pick latest by LastModified
    objs = [o for o in resp["Contents"] if o["Key"].endswith(".json")]
    if not objs:
        raise RuntimeError("Zones prefix exists but no JSON files found.")

    latest_key = sorted(objs, key=lambda x: x["LastModified"])[-1]["Key"]

    os.makedirs(TMP_DIR, exist_ok=True)
    local = os.path.join(TMP_DIR, "latest_zones.json")
    s3.download_file(S3_BUCKET, latest_key, local)

    with open(local, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload["_s3_key"] = latest_key
    return payload

def get_zones_payload():
    global _zones_payload
    if _zones_payload is None:
        _zones_payload = load_latest_zones()
    return _zones_payload

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h1>Cinematic Inference API</h1>
    <p>🚀 API ready! Documentation: <a href="/docs">/docs</a></p>
    <p>FastAPI automatic docs: <a href="/docs">Swagger UI</a></p>
    """

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/zones")
def zones():
    payload = get_zones_payload()
    return {
        "run_id": payload.get("run_id"),
        "created_at_utc": payload.get("created_at_utc"),
        "metrics": payload.get("metrics", {}),
        "zones_s3_key": payload.get("_s3_key"),
        "zones": payload.get("zones", []),
    }

@app.post("/predict")
def predict(loc: Location):
    payload = get_zones_payload()
    zones = payload.get("zones", [])

    if not zones:
        return {
            "is_hotspot": False,
            "nearest_cluster": None,
            "distance_m": None,
            "reason": "no_zones_available"
        }

    best = None
    best_dist = None

    for z in zones:
        d = haversine_m(loc.lat, loc.lon, z["lat"], z["lon"])
        if best_dist is None or d < best_dist:
            best_dist = d
            best = z

    is_hotspot = bool(best_dist <= float(best["radius_m"]))

    return {
        "is_hotspot": is_hotspot,
        "nearest_cluster": int(best["cluster"]),
        "distance_m": float(best_dist),
        "zone_radius_m": float(best["radius_m"]),
        "run_id": payload.get("run_id"),
    }