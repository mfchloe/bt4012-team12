"""Named feature lists shared by the modelling notebooks.

Columns come from data/features/<group>_<split>.parquet (see notebooks/03_feature_engineering.ipynb).
"""

# Features that separate real from fake in DGM4 because of how the dataset was built, not because of
# the manipulation itself. Exclude them from the main models; include them only in an ablation that
# shows how much a model would gain by exploiting the shortcut.
ARTIFACT_FEATURES = [
    # Absolute JPEG error levels: only informative for Guardian images, whose fakes were re-compressed
    # while the Guardian originals were not (03_feature_engineering, section 2.5)
    "img_ela_mean",
    "img_ela_face_max",
    "img_ela_face_min",
    "img_ela_face_mean",
    "img_ela_face_largest",
    # Caption length: swapped captions are ~2 words longer on average (01_data_exploration)
    "txt_word_count",
    "txt_char_count",
]


def feature_columns(df, prefix, include_artifacts=False):
    """Feature columns of a group (prefix "txt_" or "img_"), excluding artifact features by default."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not include_artifacts:
        cols = [c for c in cols if c not in ARTIFACT_FEATURES]
    return cols
