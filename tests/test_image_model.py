"""Unit tests for src/image_model.py. Run from the repo root: python -m unittest discover tests"""

import inspect
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from src.image_model import (describe_image_bytes, image_tampered_from_fake_cls, mean_face_embedding,
                             parse_fake_cls, valid_face_boxes)

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


class FaceBoxTest(unittest.TestCase):
    def test_box_inside_image_is_kept(self):
        self.assertEqual(valid_face_boxes([[10, 20, 60, 90]], width=100, height=100), [(10, 20, 60, 90)])

    def test_box_is_clipped_to_image(self):
        self.assertEqual(valid_face_boxes([[-5, -3, 120, 50]], width=100, height=40), [(0, 0, 100, 40)])

    def test_small_or_empty_boxes_are_dropped(self):
        boxes = [[0, 0, 15, 40], [0, 0, 40, 15], [50, 50, 50, 80], [90, 90, 200, 200], [0, 0, 16, 16]]
        self.assertEqual(valid_face_boxes(boxes, width=100, height=100), [(0, 0, 16, 16)])

    def test_float_and_numpy_boxes(self):
        boxes = np.array([[10.7, 20.2, 60.9, 90.1]])
        self.assertEqual(valid_face_boxes(boxes, width=100, height=100), [(10, 20, 60, 90)])

    def test_no_boxes(self):
        self.assertEqual(valid_face_boxes([], width=100, height=100), [])

    def test_crop_rule_only_sees_detector_boxes_and_image_size(self):
        # The crop rule must be the same for real and manipulated images, so it cannot take the edit box
        self.assertEqual(list(inspect.signature(valid_face_boxes).parameters), ["boxes", "width", "height", "min_side"])


class MeanFaceEmbeddingTest(unittest.TestCase):
    def test_mean_renormalised_and_fallback(self):
        faces = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
        owner = np.array([0, 0, 2])
        full = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        emb, n_faces = mean_face_embedding(faces, owner, full)

        np.testing.assert_allclose(emb[0], [2 ** -0.5, 2 ** -0.5], rtol=1e-6)  # mean of two faces, unit length
        np.testing.assert_allclose(emb[1], full[1])                           # no valid face: whole image
        np.testing.assert_allclose(emb[2], [0.6, 0.8], rtol=1e-6)             # one face: that face
        np.testing.assert_array_equal(n_faces, [2, 0, 1])
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-6)

    def test_does_not_modify_inputs(self):
        faces = np.array([[1.0, 0.0]], dtype=np.float32)
        full = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        mean_face_embedding(faces, np.array([1]), full)
        np.testing.assert_array_equal(full, [[0.0, 1.0], [0.0, 1.0]])

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mean_face_embedding(np.zeros((1, 3)), np.array([0]), np.zeros((1, 2)))


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
