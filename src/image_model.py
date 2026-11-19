"""Labels, file checks and face-crop rules for the image-only model (notebooks/08_image_model.ipynb).

The image model sees pixels only. Captions, file paths, ids and the annotation fields (`fake_image_box`,
`fake_text_pos`) are never inputs: the path alone gives the label away (untouched images live in
DGM4/origin/, edited ones in DGM4/manipulation/), and the edit box is only known for manipulated images.
"""

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image

from src.images import archive_for, open_archive

IMAGE_TAGS = {"face_swap", "face_attribute"}
TEXT_TAGS = {"text_swap", "text_attribute"}
MIN_FACE_SIDE = 16  # px; MTCNN boxes smaller than this on either side are too few pixels to embed


def parse_fake_cls(fake_cls):
    """The set of manipulation tags in a `fake_cls` value, e.g. "face_swap&text_attribute".

    Raises ValueError for anything outside the nine DGM4 combinations: "orig", or at most one image tag
    and at most one text tag.
    """
    tags = fake_cls.split("&")
    tag_set = frozenset(tags)
    if fake_cls == "orig":
        return tag_set
    # Reject unknown or empty tags ("face_swap&"), repeated tags, "orig" mixed with edits, and two image or two text tags
    unknown = tag_set - IMAGE_TAGS - TEXT_TAGS
    if unknown or len(tag_set) != len(tags) or len(tag_set & IMAGE_TAGS) > 1 or len(tag_set & TEXT_TAGS) > 1:
        raise ValueError(f"unexpected fake_cls: {fake_cls!r}")
    return tag_set


def image_tampered_from_fake_cls(fake_cls):
    """1 if the image was manipulated (face_swap or face_attribute, whatever happened to the caption), else 0."""
    return int(bool(parse_fake_cls(fake_cls) & IMAGE_TAGS))


def describe_image_bytes(data):
    """MD5 and basic properties of an encoded image. Fully decodes it, so a truncated file raises."""
    image = Image.open(BytesIO(data))  # only reads the header
    image_format = image.format
    image.load()  # decodes every pixel, which is what catches truncated or corrupt files
    return {"md5": hashlib.md5(data).hexdigest(), "width": image.width, "height": image.height,
            "mode": image.mode, "format": image_format}


def check_image_safe(image_path):
    """describe_image_bytes for a DGM4 `image` path, read from its zip. Never raises, so one bad file
    cannot stop a long run; problems are reported in `error`."""
    try:
        # "DGM4/origin/bbc/0272/120.jpg" is member "0272/120.jpg" of origin/bbc.zip (same mapping as src/images.py)
        archive, index = open_archive(archive_for(image_path))
        member = "/".join(image_path.removeprefix("DGM4/").split("/")[2:])
        if member not in index:
            return {"image": image_path, "error": "missing"}
        return {"image": image_path, "zip_member": index[member], **describe_image_bytes(archive.read(index[member])),
                "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"image": image_path, "error": repr(exc)}


def valid_face_boxes(boxes, width, height, min_side=MIN_FACE_SIDE):
    """MTCNN boxes clipped to the image, keeping those at least `min_side` pixels on both sides.

    The same rule is applied to every image, real or manipulated. It deliberately takes no edit box.
    """
    kept = []
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        # Some detector boxes extend past the image edge; keep only the part inside the picture
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 - x1 >= min_side and y2 - y1 >= min_side:
            kept.append((x1, y1, x2, y2))
    return kept


def mean_face_embedding(face_emb, owner, full_emb):
    """Average the face embeddings of each image and re-normalise to unit length.

    face_emb: (n_faces, dim) embeddings; owner: (n_faces,) index of each face's image in full_emb.
    Images without a valid face fall back to their whole-image embedding (full_emb, (n_images, dim)).
    Returns the (n_images, dim) embeddings and the number of faces averaged per image (0 = fallback).
    """
    face_emb, full_emb, owner = np.asarray(face_emb), np.asarray(full_emb), np.asarray(owner, dtype=int)
    if face_emb.shape[1:] != full_emb.shape[1:]:
        raise ValueError("face and whole-image embeddings have different dimensions")
    n_faces = np.bincount(owner, minlength=len(full_emb))  # faces per image
    sums = np.zeros(full_emb.shape, dtype=np.float64)
    np.add.at(sums, owner, face_emb)                         # add each face's embedding to its image's row
    has_face = n_faces > 0
    out = full_emb.astype(np.float32, copy=True)             # starts as the whole-image fallback for every image
    means = sums[has_face] / n_faces[has_face, None]
    # CLIP embeddings are unit length; the mean of several is shorter, so rescale it to match
    out[has_face] = means / np.linalg.norm(means, axis=1, keepdims=True)
    return out, n_faces
