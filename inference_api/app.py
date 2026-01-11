from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import boto3, pickle, os


S3_BUCKET = "jedha-lead-bucket"
MODEL_PREFIX = "models/"

def load_latest_model():
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=MODEL_PREFIX)

    if "Contents" not in resp:
        raise RuntimeError("No models found in S3. Cannot start inference API.")

    objects = resp["Contents"]
    latest = sorted(objects, key=lambda x: x["LastModified"])[-1]["Key"]

    TMP_DIR = "/tmp"

    os.makedirs(TMP_DIR, exist_ok=True)
    local = os.path.join(TMP_DIR, "latest_model.pkl")
    
    s3.download_file(S3_BUCKET, latest, local)

    with open(local, "rb") as f:
        return pickle.load(f)

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_latest_model()
    return _model

app = FastAPI(title="Cinematic Paris Hotspot API")

class Location(BaseModel):
    lat: float
    lon: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(loc: Location):
    X = np.radians([[loc.lat, loc.lon]])
    model = get_model()
    cluster = int(model.predict(X)[0])

    return {
        "cluster": cluster,
        "is_hotspot": cluster != -1
    }