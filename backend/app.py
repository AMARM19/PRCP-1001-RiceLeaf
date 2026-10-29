"""Flask REST API for rice leaf disease classification.

Endpoints
  GET  /health                 liveness + model/runtime info
  GET  /api/v1/metadata        model metrics and comparison table
  GET  /api/v1/diseases        disease descriptions and management tips
  POST /api/v1/predict         multipart field "file" (one image) or "files" (several)
                               query ?return_image=1 adds the 224x224 crop the model saw (base64 JPEG)
"""
from __future__ import annotations

import base64
import io
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, request  # noqa: E402
from PIL import UnidentifiedImageError  # noqa: E402

from inference import Predictor  # noqa: E402
from riceleaf.disease_info import DISEASE_INFO  # noqa: E402

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILES = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("riceleaf-api")


def create_app(predictor: Predictor | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024
    app.predictor = predictor or Predictor()
    log.info("Loaded model '%s' with runtime '%s'", app.predictor.metadata.get("model_name"), app.predictor.runtime)

    def error(msg, code):
        return jsonify({"error": msg}), code

    @app.get("/")
    def index():
        return jsonify({"service": "Rice Leaf Disease API", "docs": "/health, /api/v1/metadata, /api/v1/diseases, POST /api/v1/predict"})

    @app.get("/health")
    def health():
        p = app.predictor
        return jsonify({"status": "ok", "model": p.metadata.get("model_name"), "runtime": p.runtime,
                        "classes": p.class_names})

    @app.get("/api/v1/metadata")
    def metadata():
        return jsonify(app.predictor.metadata)

    @app.get("/api/v1/diseases")
    def diseases():
        return jsonify(DISEASE_INFO)

    @app.post("/api/v1/predict")
    def predict():
        uploads = request.files.getlist("files") or request.files.getlist("file")
        if not uploads:
            return error("No image uploaded. Send multipart field 'file' (or 'files').", 400)
        if len(uploads) > MAX_FILES:
            return error(f"At most {MAX_FILES} images per request.", 400)
        want_image = request.args.get("return_image", "0") in ("1", "true")

        results = []
        for f in uploads:
            name = f.filename or "upload"
            if Path(name).suffix.lower() not in ALLOWED_EXT:
                results.append({"filename": name, "error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXT)}"})
                continue
            try:
                r = app.predictor.predict(f.read())
            except (UnidentifiedImageError, OSError):
                results.append({"filename": name, "error": "File is not a valid image."})
                continue
            img = r.pop("_processed_image")
            if want_image:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                r["processed_image"] = base64.b64encode(buf.getvalue()).decode()
            r["filename"] = name
            log.info("predict %s -> %s (%.3f)", name, r["prediction"], r["confidence"])
            results.append(r)

        if "file" in request.files and len(results) == 1:
            res = results[0]
            return (jsonify(res), 422) if "error" in res else jsonify(res)
        return jsonify({"results": results})

    @app.errorhandler(413)
    def too_large(_):
        return error(f"File too large (limit {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB).", 413)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
