"""Image preprocessing shared by the notebook, the Flask API and the tests.

Keeping one implementation guarantees that the model sees exactly the same
transformation at training time and at inference time.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224
CLASS_NAMES = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]

# Background in this dataset is white/light grey (low saturation); leaf tissue
# and lesions are green/brown/yellow (higher saturation).
SATURATION_THRESHOLD = 40
CROP_MARGIN = 0.03


def load_image(source) -> Image.Image:
    """Open a path, bytes or file-like object as an RGB PIL image (EXIF-rotated)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def leaf_mask(img: Image.Image) -> np.ndarray:
    """Boolean mask of pixels that look like leaf rather than background."""
    hsv = np.asarray(img.convert("HSV"))
    return hsv[..., 1] > SATURATION_THRESHOLD


def crop_to_leaf(img: Image.Image) -> Image.Image:
    """Crop away the empty background around the leaf.

    Uses the 1st/99th percentile of mask coordinates so that stray specks do
    not blow up the bounding box. Falls back to the full image when no leaf
    pixels are found (e.g. an already tightly cropped photo).
    """
    small = img.copy()
    small.thumbnail((800, 800))
    mask = leaf_mask(small)
    if mask.mean() < 0.01:
        return img
    ys, xs = np.nonzero(mask)
    y0, y1 = np.percentile(ys, [1, 99])
    x0, x1 = np.percentile(xs, [1, 99])
    sx, sy = img.width / small.width, img.height / small.height
    mx, my = CROP_MARGIN * img.width, CROP_MARGIN * img.height
    box = (
        max(0, int(x0 * sx - mx)),
        max(0, int(y0 * sy - my)),
        min(img.width, int(x1 * sx + mx)),
        min(img.height, int(y1 * sy + my)),
    )
    if box[2] - box[0] < 10 or box[3] - box[1] < 10:
        return img
    return img.crop(box)


def preprocess(source, size: int = IMG_SIZE, crop: bool = True) -> np.ndarray:
    """Full pipeline: load -> crop to leaf -> resize -> float32 array in [0, 255].

    Scaling to the model's expected range is done inside the model itself
    (a Rescaling / preprocess_input layer), so callers never have to know it.
    """
    img = load_image(source)
    if crop:
        img = crop_to_leaf(img)
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)
