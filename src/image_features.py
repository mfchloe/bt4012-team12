"""Hand-crafted forensic features for DGM4 images.

Every feature is a fixed function of one image and its MTCNN face boxes (nothing is fitted), so it is
safe to compute the same way on every split. Functions live in a module rather than a notebook so they
can run in parallel worker processes.

Each filter is one function below. The same functions draw the example maps in 04 and compute the features,
so what the notebook shows is exactly what the models see.
"""

from io import BytesIO

import numpy as np
from PIL import Image
from scipy import ndimage

from src.images import load_image

ELA_QUALITY = 90
RING_SCALE = 0.5  # ring around each face extends this fraction of the face's width/height on every side
GAUSS_SIGMA = 2.0  # blur scale of the Gaussian high-pass residual, in pixels
PERCENTILE = 90    # used by the *_p90ratio features
# Both residuals are whole grey levels, so in a flat ring the 90th percentile can be exactly 0. Adding one grey level
# to the top and bottom of the *_p90ratio keeps such a ratio finite instead of ~1e6.
P90_FLOOR = 1.0
EPS = 1e-6

# Per-face measurements; the *_ratio versions compare the face with the ring of pixels around it
FACE_METRICS = ["ela_face", "ela_ratio", "lap_face", "lap_ratio", "noise_face", "noise_ratio"]


def grayscale(image):
    return np.asarray(image.convert("L"), dtype=np.float32)


def ela_map(image):
    # Error Level Analysis: re-save as JPEG and measure how much each pixel changes.
    # Regions pasted in or regenerated compress differently from the rest of the photo.
    rgb = np.asarray(image, dtype=np.float32)
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=ELA_QUALITY)
    resaved = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)
    return np.abs(rgb - resaved).mean(axis=2)


def laplacian_map(gray):
    # High-pass (Laplacian) residual: sharp local changes, e.g. blending seams or over-smoothed generated skin
    return np.abs(ndimage.laplace(gray))


def median_residual_map(gray):
    # Noise residual: what is left after removing the image content with a median filter.
    # A face from a different source or generator tends to carry a different noise level.
    return np.abs(gray - ndimage.median_filter(gray, size=3))


def gaussian_residual_map(gray):
    # Coarser high-pass: detail removed by a Gaussian blur, i.e. skin and hair texture a few pixels across,
    # which the 3x3 filters above miss and which regenerated or blended faces tend to lack
    return np.abs(gray - ndimage.gaussian_filter(gray, GAUSS_SIGMA))


# Every filter, for the example figures in 04: title, the function (taking the RGB image) and a plain-English meaning
FILTERS = {
    "ela": {"title": "JPEG error level analysis (ELA)", "map": ela_map,
            "meaning": "Bright = changed a lot when the photo was re-saved. A pasted-in face can stand out."},
    "lap": {"title": "Laplacian response", "map": lambda image: laplacian_map(grayscale(image)),
            "meaning": "Bright = sharp edges and fine detail. An edited face often looks darker (smoother) than the area around it."},
    "noise": {"title": "Median-filter residual", "map": lambda image: median_residual_map(grayscale(image)),
              "meaning": "Bright = fine noise. A face taken from another photo can have a different noise level."},
    "gauss": {"title": f"Gaussian high-pass residual (sigma={GAUSS_SIGMA:g})",
              "map": lambda image: gaussian_residual_map(grayscale(image)),
              "meaning": "Bright = skin and hair texture. A computer-generated face often looks flatter."},
}


def forensic_maps(image):
    """Per-pixel maps that highlight editing traces (the `image` feature group)."""
    gray = grayscale(image)
    return {"ela": ela_map(image), "lap": laplacian_map(gray), "noise": median_residual_map(gray)}


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    return max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))


def face_regions(box, width, height):
    """The face box and the outer box of its ring, both clipped to the image, or None if the face is too small.
    The ring is the outer box minus the face."""
    x1, y1, x2, y2 = clip_box(box, width, height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    pad_x, pad_y = int(RING_SCALE * (x2 - x1)), int(RING_SCALE * (y2 - y1))
    return (x1, y1, x2, y2), clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def face_measurements(maps, box, width, height):
    regions = face_regions(box, width, height)
    if regions is None:
        return None
    (x1, y1, x2, y2), (ox1, oy1, ox2, oy2) = regions
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


# The `image_filters` group: each per-face measurement and how it is summarised across the faces of one image
FILTER_FEATURES = {
    "lap_p90ratio": ["min", "max", "largest"],
    "noise_p90ratio": ["min", "max", "largest"],
    "gauss_ratio": ["min", "max", "largest", "spread"],
}
FILTER_COLUMNS = [f"img_{metric}_{stat}" for metric, stats in FILTER_FEATURES.items() for stat in stats]


def filter_face_measurements(maps, box, width, height):
    """Face-vs-ring comparisons that complement face_measurements:
    - lap/noise_p90ratio: (90th percentile inside the face + 1) / (90th percentile in the ring + 1). A mean averages
      away the few sharpest pixels (eyelashes, hair edges), which are what a smoothed or regenerated face loses.
    - gauss_ratio: mean Gaussian high-pass residual inside the face / in the ring, i.e. texture at a coarser scale.
    """
    regions = face_regions(box, width, height)
    if regions is None:
        return None
    (x1, y1, x2, y2), (ox1, oy1, ox2, oy2) = regions
    ring_mask = np.ones((oy2 - oy1, ox2 - ox1), dtype=bool)
    ring_mask[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = False

    result = {"rel_area": (x2 - x1) * (y2 - y1) / (width * height)}
    for name, m in maps.items():
        face, ring = m[y1:y2, x1:x2], m[oy1:oy2, ox1:ox2][ring_mask]
        if name == "gauss":
            metric, face_value, floor = "gauss_ratio", face.mean(), EPS
            ring_value = ring.mean() if ring.size else None
        else:
            metric, face_value, floor = f"{name}_p90ratio", np.percentile(face, PERCENTILE), P90_FLOOR
            ring_value = np.percentile(ring, PERCENTILE) if ring.size else None
        # Same convention as face_measurements: a face with no ring around it (it fills the photo) gets ratio 1
        result[metric] = 1.0 if ring_value is None else (face_value + floor) / (ring_value + floor)
    return result


def filter_features(image_path, boxes):
    image = load_image(image_path)
    width, height = image.size
    gray = grayscale(image)
    maps = {"lap": laplacian_map(gray), "noise": median_residual_map(gray), "gauss": gaussian_residual_map(gray)}

    # No usable face: every column stays missing (NaN), and the model pipeline fills it with the training median
    features = {c: np.nan for c in FILTER_COLUMNS}
    faces = [f for f in (filter_face_measurements(maps, box, width, height) for box in boxes) if f is not None]
    if not faces:
        return features
    largest = faces[int(np.argmax([f["rel_area"] for f in faces]))]
    for metric, stats in FILTER_FEATURES.items():
        values = np.array([f[metric] for f in faces])
        summary = {"min": values.min(), "max": values.max(), "largest": largest[metric],
                   "spread": np.log(values.max()) - np.log(values.min())}
        for stat in stats:
            features[f"img_{metric}_{stat}"] = summary[stat]
    return features


def filter_features_safe(item):
    """Parallel-map wrapper for filter_features, like image_features_safe."""
    image_path, boxes = item
    try:
        return {"image": image_path, **filter_features(image_path, boxes), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"image": image_path, "error": repr(exc)}
