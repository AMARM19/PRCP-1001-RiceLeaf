"""Streamlit UI for the Rice Leaf Disease API."""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
from PIL import Image

DEFAULT_API = os.getenv("BACKEND_URL", "http://localhost:5000").rstrip("/")
SAMPLE_DIR = Path(__file__).parent / "samples"   # held-out test images, one per class

st.set_page_config(page_title="Rice Leaf Doctor", page_icon="🌾", layout="wide")


@st.cache_data(ttl=30, show_spinner=False)
def api_get(base: str, path: str):
    r = requests.get(f"{base}{path}", timeout=60)  # generous: free hosts cold-start slowly
    r.raise_for_status()
    return r.json()


@dataclass
class ImageFile:
    name: str
    data: bytes


def api_predict(base: str, files: list[ImageFile]) -> list[dict]:
    payload = [("files", (f.name, f.data, "application/octet-stream")) for f in files]
    r = requests.post(f"{base}/api/v1/predict", params={"return_image": 1}, files=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(r.json().get("error", r.text))
    return r.json()["results"]


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🌾 Rice Leaf Doctor")
    api = st.text_input("Backend API URL", DEFAULT_API).rstrip("/")
    try:
        health = api_get(api, "/health")
        st.success(f"API online · {health['runtime']}")
        st.caption(f"Model: **{health['model']}**")
    except Exception as e:  # noqa: BLE001
        health = None
        st.error("API unreachable")
        st.caption(str(e)[:200])
    st.divider()
    st.caption("Detects **Bacterial leaf blight**, **Brown spot** and **Leaf smut** from a photo of a single rice leaf.")
    st.caption("Best results: one leaf, plain light background, lesions in focus.")

tab_predict, tab_report, tab_about = st.tabs(["🔍 Diagnose", "📊 Model report", "ℹ️ Diseases & data"])

# ---------------------------------------------------------------- diagnose
with tab_predict:
    st.header("Diagnose a rice leaf")
    c1, c2 = st.columns([3, 2])
    with c1:
        uploads = st.file_uploader("Upload leaf image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                   accept_multiple_files=True)
    with c2:
        use_cam = st.toggle("Use camera instead")
        cam = st.camera_input("Take a photo") if use_cam else None
    samples = sorted(SAMPLE_DIR.glob("*.jpg"))
    picked = st.multiselect("…or try sample image(s) from the held-out test set", samples,
                            format_func=lambda p: p.stem.rsplit("_", 1)[0].replace("_", " ").capitalize())
    files = [ImageFile(f.name, f.getvalue()) for f in uploads or []]
    if cam is not None:
        files.append(ImageFile("camera.jpg", cam.getvalue()))
    files += [ImageFile(p.name, p.read_bytes()) for p in picked]

    if files and st.button(f"Analyse {len(files)} image(s)", type="primary", disabled=health is None):
        with st.spinner("Running model…"):
            try:
                st.session_state["results"] = list(zip(files, api_predict(api, files[:10])))
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")

    for f, r in st.session_state.get("results", []):
        st.divider()
        if "error" in r:
            st.error(f"**{r['filename']}**: {r['error']}")
            continue
        a, b, c = st.columns([3, 1.3, 3])
        a.image(Image.open(io.BytesIO(f.data)), caption=r["filename"], width="stretch")
        if "processed_image" in r:
            b.image(base64.b64decode(r["processed_image"]), caption="What the model sees", width="stretch")
        with c:
            label, conf = r["prediction"], r["confidence"]
            st.subheader(label)
            st.metric("Confidence", f"{conf:.1%}", help=f"Inference {r['latency_ms']} ms")
            if r["low_confidence"]:
                st.warning("Low confidence – the image may be unclear, show several diseases, or be out of scope "
                           "(e.g. healthy leaf / other disease). Please retake the photo or consult an expert.")
            probs = pd.DataFrame({"class": list(r["probabilities"]), "probability": list(r["probabilities"].values())})
            st.bar_chart(probs, x="class", y="probability", color="#2a9d8f", horizontal=True, height=160)
        info = r.get("disease_info") or {}
        if info:
            with st.expander(f"About {label} & what to do", expanded=not r["low_confidence"]):
                st.markdown(f"**Pathogen:** {info['pathogen']}\n\n**Symptoms:** {info['symptoms']}")
                st.markdown("**Management:**\n" + "\n".join(f"- {m}" for m in info["management"]))

# ---------------------------------------------------------------- report
with tab_report:
    st.header("Model performance")
    if health is None:
        st.info("Connect to the API to load the model report.")
    else:
        meta = api_get(api, "/api/v1/metadata")
        st.markdown(f"**Production model:** `{meta.get('model_name', '–')}`")
        m2, m3, m4 = st.columns(3)
        m2.metric("Test accuracy", f"{meta.get('test_accuracy', 0):.1%}")
        m3.metric("Test macro-F1", f"{meta.get('test_macro_f1', 0):.3f}")
        m4.metric("Validation accuracy", f"{meta.get('val_accuracy', 0):.1%}")
        n = meta.get("n_images", {})
        st.caption(f"Split: {n.get('train')} train / {n.get('val')} validation / {n.get('test')} test images. "
                   "With only 18 test images one mistake ≈ 5.6 percentage points, so read the numbers with care.")

        comp = pd.DataFrame(meta.get("comparison", []))
        if not comp.empty:
            st.subheader("All models compared")
            st.caption("Bars = test accuracy (grey = baseline, green = improved). Black tick = 5-fold CV accuracy "
                       "(classical models) or validation accuracy (CNNs), which is the more stable signal.")
            comp["check_acc"] = comp.cv_acc.fillna(comp.val_acc)
            order = comp.sort_values(["test_acc", "check_acc"], ascending=False).model.tolist()
            y = alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
            bars = alt.Chart(comp).mark_bar().encode(
                x=alt.X("test_acc:Q", title="accuracy", scale=alt.Scale(domain=[0, 1])), y=y,
                color=alt.Color("stage:N", scale=alt.Scale(domain=["Baseline", "Improved"], range=["#9e9e9e", "#2a9d8f"])),
                tooltip=["model", "stage", alt.Tooltip("test_acc", format=".3f"), alt.Tooltip("check_acc", format=".3f")])
            ticks = alt.Chart(comp).mark_tick(color="black", thickness=2, size=14).encode(x="check_acc:Q", y=y)
            st.altair_chart(bars + ticks, width="stretch")
            show = comp[["model", "stage", "features", "cv_acc", "val_acc", "test_acc", "test_f1", "params", "ms_per_img"]]
            st.dataframe(show, hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="%.3f")
                                        for k in ["cv_acc", "val_acc", "test_acc", "test_f1"]}
                         | {"params": st.column_config.NumberColumn(format="%d"),
                            "ms_per_img": st.column_config.NumberColumn("ms / image", format="%.1f")})

        cm, rep = meta.get("confusion_matrix"), meta.get("classification_report")
        if cm:
            l, r = st.columns(2)
            names = meta["class_names"]
            with l:
                st.subheader("Confusion matrix (test)")
                st.dataframe(pd.DataFrame(cm, index=[f"true: {c}" for c in names], columns=[f"pred: {c}" for c in names]),
                             width="stretch")
            with r:
                st.subheader("Per-class metrics (test)")
                st.dataframe(pd.DataFrame({c: rep[c] for c in names}).T[["precision", "recall", "f1-score", "support"]]
                             .round(3), width="stretch")

# ---------------------------------------------------------------- about
with tab_about:
    st.header("The three diseases")
    try:
        diseases = api_get(api, "/api/v1/diseases")
    except Exception:  # noqa: BLE001
        diseases = {}
    cols = st.columns(3)
    for col, (name, info) in zip(cols, diseases.items()):
        with col:
            st.markdown(f"### {name}")
            st.markdown(f"**Pathogen:** {info['pathogen']}")
            st.markdown(f"**Symptoms:** {info['symptoms']}")
            st.markdown("**Management:**\n" + "\n".join(f"- {m}" for m in info["management"]))
    st.divider()
    st.subheader("About the data & model")
    st.markdown("""
- **Dataset:** 119 photos of infected rice leaves (40 bacterial leaf blight, 40 brown spot, 39 leaf smut).
- **Pipeline:** crop to leaf → resize 224×224 → CNN (ImageNet transfer learning) trained with data augmentation.
- **Limits:** the model only knows these 3 diseases. A healthy leaf or another disease (e.g. leaf blast) will
  still be assigned one of the three classes, so watch the low-confidence warning.
""")
