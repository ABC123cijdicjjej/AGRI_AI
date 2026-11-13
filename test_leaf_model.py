import unittest

import numpy as np
from PIL import Image

from leaf_model import (
    LABELS,
    _normalize_legacy_sequential_config,
    display_label,
    load_model,
    predict_image,
)


class FakeModel:
    def __init__(self, scores):
        self.scores = np.asarray([scores], dtype=np.float32)

    def predict(self, batch, verbose=0):
        if batch.shape != (1, 128, 128, 3):
            raise AssertionError("Unexpected model input shape: {}".format(batch.shape))
        return self.scores


class LeafModelTests(unittest.TestCase):
    def test_preprocessing_and_top_three_labels(self):
        scores = np.zeros(len(LABELS), dtype=np.float32)
        scores[4] = 0.73
        scores[0] = 0.18
        scores[10] = 0.06

        ranked = predict_image(
            Image.new("RGB", (240, 160), color=(30, 120, 45)),
            model=FakeModel(scores),
        )

        self.assertEqual([label for label, _ in ranked], [
            LABELS[4], LABELS[0], LABELS[10],
        ])
        self.assertAlmostEqual(ranked[0][1], 0.73, places=5)

    def test_rejects_output_with_wrong_number_of_classes(self):
        with self.assertRaisesRegex(ValueError, "returned 2 classes"):
            predict_image(
                Image.new("RGB", (128, 128)),
                model=FakeModel([0.6, 0.4]),
            )

    def test_class_names_are_readable(self):
        self.assertEqual(display_label("Corn_(maize)"), "Corn (Maize)")
        self.assertEqual(display_label("Common_rust_"), "Common Rust")

    def test_normalizes_legacy_sequential_model_config(self):
        architecture = {
            "class_name": "Sequential",
            "config": [
                {
                    "class_name": "Conv2D",
                    "config": {
                        "name": "conv2d",
                        "batch_input_shape": [None, 128, 128, 3],
                        "dtype": "float32",
                    },
                },
            ],
        }

        normalized = _normalize_legacy_sequential_config(architecture)

        self.assertEqual(normalized["config"]["name"], "plant_leaf_cnn")
        self.assertEqual(normalized["config"]["layers"][0]["class_name"], "InputLayer")
        self.assertEqual(
            normalized["config"]["layers"][0]["config"]["batch_shape"],
            [None, 128, 128, 3],
        )
        self.assertNotIn(
            "batch_input_shape",
            normalized["config"]["layers"][1]["config"],
        )

    def test_loads_saved_model_and_weights(self):
        model = load_model()
        self.assertEqual(model.output_shape[-1], len(LABELS))


if __name__ == "__main__":
    unittest.main()
