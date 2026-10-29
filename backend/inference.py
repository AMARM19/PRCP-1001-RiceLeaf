"""Model loading and prediction for the Flask API.

Runtime preference (override with MODEL_RUNTIME=litert|tflite|keras):
  1. LiteRT (`ai-edge-litert`)  - tiny dependency, used in the Docker image
  2. tf.lite from TensorFlow    - local development
  3. Full Keras model           - fallback when no .tflite file is present
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from riceleaf.disease_info import DISEASE_INFO
from riceleaf.preprocess import CLASS_NAMES, IMG_SIZE, crop_to_leaf, load_image

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.6"))


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR, runtime: str | None = None):
        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / "model_metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.class_names = self.metadata.get("class_names", CLASS_NAMES)
        self.img_size = self.metadata.get("img_size", IMG_SIZE)
        self._lock = threading.Lock()  # TFLite interpreters are not thread-safe
        self.runtime = self._load(runtime or os.getenv("MODEL_RUNTIME", "auto"))

    def _load(self, runtime: str) -> str:
        tflite_path = self.model_dir / "rice_leaf_model.tflite"
        keras_path = self.model_dir / "rice_leaf_model.keras"
        if runtime in ("auto", "litert") and tflite_path.exists():
            try:
                from ai_edge_litert.interpreter import Interpreter
                return self._init_interpreter(Interpreter, tflite_path, "litert")
            except ImportError:
                if runtime == "litert":
                    raise
        if runtime in ("auto", "tflite") and tflite_path.exists():
            try:
                import tensorflow as tf
                return self._init_interpreter(tf.lite.Interpreter, tflite_path, "tflite")
            except ImportError:
                if runtime == "tflite":
                    raise
        if runtime in ("auto", "keras") and keras_path.exists():
            import keras
            self._keras = keras.models.load_model(keras_path)
            return "keras"
        raise FileNotFoundError(f"No usable model found in {self.model_dir} (runtime={runtime})")

    def _init_interpreter(self, cls, path: Path, name: str) -> str:
        self._interp = cls(model_path=str(path))
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]["index"]
        self._out = self._interp.get_output_details()[0]["index"]
        return name

    def _forward(self, x: np.ndarray) -> np.ndarray:
        if self.runtime == "keras":
            return self._keras.predict(x[None], verbose=0)[0]
        with self._lock:
            self._interp.set_tensor(self._in, x[None].astype(np.float32))
            self._interp.invoke()
            return self._interp.get_tensor(self._out)[0].copy()

    def predict(self, source) -> dict:
        t0 = time.perf_counter()
        img = crop_to_leaf(load_image(source)).resize((self.img_size, self.img_size))
        x = np.asarray(img, dtype=np.float32)
        probs = self._forward(x)
        top = int(np.argmax(probs))
        label = self.class_names[top]
        conf = float(probs[top])
        return {
            "prediction": label,
            "confidence": round(conf, 4),
            "low_confidence": conf < LOW_CONFIDENCE,
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.class_names, probs)},
            "disease_info": DISEASE_INFO.get(label, {}),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "_processed_image": img,
        }
