"""Labels, file checks, input rules and classifiers for the image-only model (notebooks/07_image_model.ipynb).

The image model sees pixels only: hand-crafted forensic features (`img_*`, notebook 04) and, in its second step,
CLIP image embeddings (`clip_img_*`, notebook 05). Captions, caption embeddings, image-caption similarities, file
paths, ids and the annotation fields (`fake_image_box`, `fake_text_pos`) are never inputs: the path alone gives the
label away (untouched images live in DGM4/origin/, edited ones in DGM4/manipulation/), and the edit box is only
known for manipulated images.
"""

import hashlib
from io import BytesIO

from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.feature_sets import ARTIFACT_FEATURES
from src.images import archive_for, open_archive

IMAGE_TAGS = {"face_swap", "face_attribute"}
TEXT_TAGS = {"text_swap", "text_attribute"}
RANDOM_STATE = 4012
# Column prefixes the image model may read: forensic features and CLIP *image* embeddings. Not clip_txt_* (the caption),
# and not the clip_sim / clip_face_* groups, which compare the image with its caption.
INPUT_PREFIXES = ("img_", "clip_img_")


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


def check_model_inputs(columns):
    """Raise if any column is not an allowed image input, or is one of the known shortcut features."""
    bad = [c for c in columns if not c.startswith(INPUT_PREFIXES) or c in ARTIFACT_FEATURES]
    if bad:
        raise ValueError(f"not allowed as image-model inputs: {bad}")
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate input columns")
    return list(columns)


def make_l1(C):
    """Lasso logistic regression used to prune the engineered features: weak or duplicate features get weight 0."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         # l1_ratio=1 is the pure L1 (lasso) penalty; sklearn >= 1.8 spells it this way instead of penalty="l1"
                         LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, random_state=RANDOM_STATE))


def make_classifiers():
    """The two image-only classifiers, with fixed settings (no tuning). The imputer and scaler are part of each
    pipeline, so they are fit on the training rows only. Only logistic regression needs scaled inputs."""
    return {
        "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "XGBoost": make_pipeline(SimpleImputer(strategy="median"),
                                 XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                                               random_state=RANDOM_STATE, n_jobs=4)),
    }


def select_baseline(val_aucs, decimals=3):
    """The classifier with the higher validation AUC; Logistic Regression on a tie (it is the simpler model).
    AUCs are compared at the precision they are reported (3 decimals), so a 0.0001 gap counts as a tie."""
    lr, xgb = round(val_aucs["Logistic Regression"], decimals), round(val_aucs["XGBoost"], decimals)
    return "XGBoost" if xgb > lr else "Logistic Regression"
