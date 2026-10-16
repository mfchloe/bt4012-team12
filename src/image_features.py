"""Hand-crafted forensic features for DGM4 images.

Every feature is a fixed function of one image and its MTCNN face boxes (nothing is fitted), so it is
safe to compute the same way on every split. Functions live in a module rather than a notebook so they
can run in parallel worker processes.
"""

from io import BytesIO

import numpy as np
from PIL import Image
from scipy import ndimage

from src.images import load_image

ELA_QUALITY = 90
RING_SCALE = 0.5  # ring around each face extends this fraction of the face's width/height on every side
EPS = 1e-6

# Per-face measurements; the *_ratio versions compare the face with the ring of pixels around it
FACE_METRICS = ["ela_face", "ela_ratio", "lap_face", "lap_ratio", "noise_face", "noise_ratio"]


def forensic_maps(image):
    """Per-pixel maps that highlight editing traces."""
    rgb = np.asarray(image, dtype=np.float32)
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    # Error Level Analysis: re-save as JPEG and measure how much each pixel changes.
    # Regions pasted in or regenerated compress differently from the rest of the photo.
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=ELA_QUALITY)
    resaved = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)
    ela = np.abs(rgb - resaved).mean(axis=2)

    # High-pass (Laplacian) residual: sharp local changes, e.g. blending seams or over-smoothed generated skin
    lap = np.abs(ndimage.laplace(gray))

    # Noise residual: what is left after removing the image content with a median filter.
    # A face from a different source or generator tends to carry a different noise level.
    noise = np.abs(gray - ndimage.median_filter(gray, size=3))

    return {"ela": ela, "lap": lap, "noise": noise}


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    return max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))


def face_measurements(maps, box, width, height):
    x1, y1, x2, y2 = clip_box(box, width, height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    pad_x, pad_y = int(RING_SCALE * (x2 - x1)), int(RING_SCALE * (y2 - y1))
    ox1, oy1, ox2, oy2 = clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)
    face_area = (x2 - x1) * (y2 - y1)
    ring_area = (ox2 - ox1) * (oy2 - oy1) - face_area

    result = {"rel_area": face_area / (width * height)}
    for name, m in maps.items():
        face_sum = m[y1:y2, x1:x2].sum()
        face_mean = face_sum / face_area
        if ring_area > 0:
            ring_mean = (m[oy1:oy2, ox1:ox2].sum() - face_sum) / ring_area
            ratio = (face_mean + EPS) / (ring_mean + EPS)
        else:
            ratio = 1.0
        result[f"{name}_face"] = face_mean
        result[f"{name}_ratio"] = ratio
    return result


def image_features(image_path, boxes):
    image = load_image(image_path)
    width, height = image.size
    maps = forensic_maps(image)

    features = {
        "img_width": width,
        "img_height": height,
        "img_n_faces": len(boxes),
        **{f"img_{name}_mean": m.mean() for name, m in maps.items()},
    }

    faces = [f for f in (face_measurements(maps, box, width, height) for box in boxes) if f is not None]
    if not faces:
        return features

    areas = np.array([f["rel_area"] for f in faces])
    largest = faces[int(areas.argmax())]
    features["img_face_area_max"] = areas.max()
    features["img_face_area_mean"] = areas.mean()

    # Any face may be the edited one (only ~2/3 of the time is it the largest), so summarise across all faces
    for metric in FACE_METRICS:
        values = np.array([f[metric] for f in faces])
        features[f"img_{metric}_max"] = values.max()
        features[f"img_{metric}_min"] = values.min()
        features[f"img_{metric}_mean"] = values.mean()
        features[f"img_{metric}_largest"] = largest[metric]
        if metric.endswith("_ratio"):
            # Every image has >= 2 faces, so one edited face should stand out from the others in the same photo
            logs = np.log(values)
            features[f"img_{metric}_spread"] = logs.max() - logs.min()
    return features


def image_features_safe(item):
    """Wrapper for parallel maps: never raises, so one bad image cannot kill a long run."""
    image_path, boxes = item
    try:
        return {"image": image_path, **image_features(image_path, boxes), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"image": image_path, "error": repr(exc)}
