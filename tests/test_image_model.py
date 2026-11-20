"""Unit tests for src/image_model.py. Run from the repo root: python -m unittest discover tests"""

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from src.image_model import (check_model_inputs, describe_image_bytes, image_tampered_from_fake_cls, make_classifiers,
                             parse_fake_cls, select_baseline)

# The nine fake_cls values in DGM4 and whether the image itself was manipulated
EXPECTED_LABELS = {
    "orig": 0,
    "text_swap": 0,
    "text_attribute": 0,
    "face_swap": 1,
    "face_attribute": 1,
    "face_swap&text_swap": 1,
    "face_swap&text_attribute": 1,
    "face_attribute&text_swap": 1,
    "face_attribute&text_attribute": 1,
}


class ImageLabelTest(unittest.TestCase):
    def test_all_nine_combinations(self):
        for fake_cls, expected in EXPECTED_LABELS.items():
            with self.subTest(fake_cls=fake_cls):
                self.assertEqual(image_tampered_from_fake_cls(fake_cls), expected)

    def test_tag_order_does_not_matter(self):
        self.assertEqual(image_tampered_from_fake_cls("text_swap&face_swap"), 1)
        self.assertEqual(parse_fake_cls("text_attribute&face_attribute"),
                         frozenset({"face_attribute", "text_attribute"}))

    def test_invalid_values_raise(self):
        for bad in ["", "&", "face_swap&", "orig&face_swap", "face", "faceswap", "face_swap&face_attribute",
                    "text_swap&text_attribute", "face_swap&face_swap", "FACE_SWAP"]:
            with self.subTest(fake_cls=bad):
                with self.assertRaises(ValueError):
                    image_tampered_from_fake_cls(bad)


class ModelInputsTest(unittest.TestCase):
    def test_forensic_features_and_image_embeddings_are_allowed(self):
        cols = ["img_lap_ratio_min", "img_gauss_ratio_max", "clip_img_0", "clip_img_767"]
        self.assertEqual(check_model_inputs(cols), cols)

    def test_caption_ids_labels_and_shortcuts_are_rejected(self):
        for bad in ["clip_txt_0", "clip_sim", "clip_face_sim_max", "txt_word_count", "row_id", "id", "image",
                    "text", "image_fake", "fake_cls", "fake_image_box", "img_ela_mean", "img_ela_face_max"]:
            with self.subTest(column=bad):
                with self.assertRaises(ValueError):
                    check_model_inputs(["img_lap_ratio_min", bad])

    def test_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            check_model_inputs(["img_lap_ratio_min", "img_lap_ratio_min"])


class SelectBaselineTest(unittest.TestCase):
    def test_higher_auc_wins(self):
        self.assertEqual(select_baseline({"Logistic Regression": 0.70, "XGBoost": 0.72}), "XGBoost")
        self.assertEqual(select_baseline({"Logistic Regression": 0.75, "XGBoost": 0.72}), "Logistic Regression")

    def test_tie_prefers_logistic_regression(self):
        self.assertEqual(select_baseline({"Logistic Regression": 0.7, "XGBoost": 0.7}), "Logistic Regression")
        self.assertEqual(select_baseline({"Logistic Regression": 0.70011, "XGBoost": 0.70014}), "Logistic Regression")


class ClassifiersTest(unittest.TestCase):
    def test_both_fit_with_missing_values_and_are_reproducible(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
        X[::7, 1] = np.nan  # e.g. an image without a usable face
        for name in ["Logistic Regression", "XGBoost"]:
            with self.subTest(model=name):
                p1 = make_classifiers()[name].fit(X, y).predict_proba(X)[:, 1]
                p2 = make_classifiers()[name].fit(X, y).predict_proba(X)[:, 1]
                np.testing.assert_array_equal(p1, p2)
                self.assertTrue(np.isfinite(p1).all())


class DescribeImageBytesTest(unittest.TestCase):
    def test_valid_jpeg(self):
        buffer = BytesIO()
        Image.new("RGB", (32, 24), (200, 10, 10)).save(buffer, "JPEG")
        info = describe_image_bytes(buffer.getvalue())
        self.assertEqual((info["width"], info["height"], info["mode"], info["format"]), (32, 24, "RGB", "JPEG"))
        self.assertEqual(len(info["md5"]), 32)

    def test_truncated_jpeg_raises(self):
        buffer = BytesIO()
        Image.new("RGB", (64, 64), (200, 10, 10)).save(buffer, "JPEG")
        with self.assertRaises(Exception):
            describe_image_bytes(buffer.getvalue()[:100])


if __name__ == "__main__":
    unittest.main()
