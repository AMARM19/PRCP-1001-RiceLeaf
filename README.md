# PRCP-1001 · Rice Leaf Disease Detection

Classifies a photo of a rice leaf as **Bacterial leaf blight**, **Brown spot** or **Leaf smut**.
The project has the full analysis notebook, a **Flask** REST API, a **Streamlit** UI, and Docker/Render deployment.

## Results at a glance

| Model | Stage | 5-fold CV | Val acc | Test acc (18 imgs) |
|---|---|---|---|---|
| SVM-RBF, hand-crafted features | Baseline | 0.851 | – | 1.000 |
| Simple CNN, no augmentation | Baseline | – | 0.778 | 0.778 |
| Simple CNN + augmentation | Improved | – | 0.944 | 0.944 |
| MobileNetV2 embeddings + LogReg | Improved | 0.891 | – | 1.000 |
| MobileNetV2 frozen + augmentation | Improved | **0.932 ± 0.035** | 0.944 | 1.000 |
| **MobileNetV2 fine-tuned + augmentation** (production) | Improved | – | **1.000** | **1.000** |
| EfficientNetB0 frozen + augmentation | Improved | – | 0.944 | 1.000 |

The test set has only 18 images and saturates. **Expect ~93% accuracy on similar photos**, based on 5-fold CV of the
production architecture. The production model ships as a 2.5 MB quantised TFLite file (~60 ms per image on CPU), and it agrees
with the Keras model on 100% of test images.

Key findings (details and the challenges report are in notebook §9–§11):
- **Image size leaks the label.** All bacterial-blight photos are 3081×897, while ~40% of the others are small crops.
  Cropping to the leaf and resizing to a fixed size removes this shortcut.
- **Augmentation adds +17 points to a from-scratch CNN** and cuts MobileNetV2's train/val gap from 0.17 to 0.02.
- **Transfer learning is the biggest lever:** a linear model on ImageNet features beats every tuned hand-crafted model.
- The brief mentions *leaf blast*, but the data contains *leaf smut*, so the model covers the three classes provided.

## Project layout

```
Data/                                raw dataset (as provided)
notebooks/RiceLeaf_Disease_Detection.ipynb
                                     single submission notebook: cleaning → EDA → preprocessing →
                                     feature engineering → baseline → improved models → reports
src/riceleaf/                        shared code (preprocessing, disease info) used by notebook AND API
models/                              exported model (.keras, .tflite), metadata, comparison CSV
reports/figures/                     key charts saved by the notebook
data_clean/                          cleaned, renamed dataset + manifest.csv (re-created by the notebook)
backend/                             Flask API (app.py, inference.py, Dockerfile)
frontend/                            Streamlit UI (app.py, Dockerfile)
tests/test_api.py                    API tests (pytest)
docker-compose.yml, render.yaml      deployment
```

## 1. Re-run the notebook

```bash
pip install tensorflow==2.20.0 scikit-learn scikit-image seaborn pandas pillow jupyter
```

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/RiceLeaf_Disease_Detection.ipynb
```

This takes about 15 minutes on CPU. It writes `models/rice_leaf_model.{keras,tflite}` and `models/model_metadata.json`.
On native Windows TensorFlow ≥ 2.11 runs on CPU only. To use an NVIDIA GPU, run the notebook under WSL2, Linux or Colab.

## 2. Run locally (no Docker)

Terminal 1, the API on http://localhost:5000:

```bash
pip install -r backend/requirements.txt
```

```bash
python backend/app.py
```

Terminal 2, the UI on http://localhost:8501:

```bash
pip install -r frontend/requirements.txt
```

```bash
streamlit run frontend/app.py
```

Tests:

```bash
python -m pytest tests -q
```

## 3. Run with Docker

```bash
docker compose up --build
```

UI → http://localhost:8501 · API → http://localhost:5000/health

## 4. Deploy to Render (free tier)

1. Push this folder to a GitHub repository. Include `models/rice_leaf_model.tflite` and `models/model_metadata.json`.
2. On Render: **New → Blueprint**, then pick the repo. `render.yaml` creates two Docker web services:
   `riceleaf-api` (Flask + gunicorn + LiteRT) and `riceleaf-ui` (Streamlit).
3. If Render gives the API a different URL, update `BACKEND_URL` on `riceleaf-ui` to match it.
4. Free services sleep after 15 minutes idle, so the first request takes about 30–60 seconds.

The API image uses the **TFLite model with `ai-edge-litert`**, not full TensorFlow. That keeps the image around 200 MB
and memory well under the free tier's 512 MB.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | status, model name, runtime |
| GET | `/api/v1/metadata` | metrics, confusion matrix, model comparison |
| GET | `/api/v1/diseases` | pathogen / symptoms / management per disease |
| POST | `/api/v1/predict` | multipart `file` (one image) or `files` (up to 10); `?return_image=1` adds the 224×224 crop |

```bash
curl -F "file=@Data/Brown spot-20200814T055208Z-001/Brown spot/DSC_0100.jpg" http://localhost:5000/api/v1/predict
```

Example response:

```json
{"prediction": "Brown spot", "confidence": 0.97, "low_confidence": false,
 "probabilities": {"Bacterial leaf blight": 0.01, "Brown spot": 0.97, "Leaf smut": 0.02},
 "disease_info": {"pathogen": "...", "symptoms": "...", "management": ["..."]},
 "latency_ms": 41.2, "filename": "DSC_0100.jpg"}
```
