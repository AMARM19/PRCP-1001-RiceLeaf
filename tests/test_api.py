import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "src")]

from app import create_app  # noqa: E402
from inference import Predictor  # noqa: E402

SAMPLES = {
    "Bacterial leaf blight": next((ROOT / "Data").glob("Bacterial*/*/*")),
    "Brown spot": next((ROOT / "Data").glob("Brown*/*/*")),
    "Leaf smut": next((ROOT / "Data").glob("Leaf*/*/*")),
}


@pytest.fixture(scope="module", params=["litert", "keras"])
def client(request):
    return create_app(Predictor(runtime=request.param)).test_client()


def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
    assert len(body["classes"]) == 3


def test_metadata_and_diseases(client):
    assert "comparison" in client.get("/api/v1/metadata").get_json()
    assert set(client.get("/api/v1/diseases").get_json()) == set(SAMPLES)


@pytest.mark.parametrize("label", list(SAMPLES))
def test_predict_single(client, label):
    path = SAMPLES[label]
    r = client.post("/api/v1/predict", data={"file": (io.BytesIO(path.read_bytes()), path.name)},
                    content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200, body
    assert body["prediction"] in SAMPLES
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["disease_info"]["management"]


def test_predict_batch_with_image(client):
    files = [(io.BytesIO(p.read_bytes()), p.name) for p in SAMPLES.values()]
    r = client.post("/api/v1/predict?return_image=1", data={"files": files}, content_type="multipart/form-data")
    results = r.get_json()["results"]
    assert len(results) == 3 and all("processed_image" in x for x in results)


def test_rejects_bad_input(client):
    assert client.post("/api/v1/predict").status_code == 400
    r = client.post("/api/v1/predict", data={"file": (io.BytesIO(b"hello"), "notes.txt")}, content_type="multipart/form-data")
    assert r.status_code == 422
    r = client.post("/api/v1/predict", data={"file": (io.BytesIO(b"not an image"), "fake.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 422 and "valid image" in r.get_json()["error"]
