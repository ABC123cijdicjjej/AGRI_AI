"""Shared model loading and image inference for the desktop and web apps."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


LABELS = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___Early_blight",
    "Potato___Healthy",
    "Potato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
)

ROOT_DIR = Path(__file__).resolve().parent
MODEL = None


def load_model():
    """Load the saved CNN once, on the first prediction."""
    global MODEL
    if MODEL is not None:
        return MODEL

    architecture_path = ROOT_DIR / "model1.json"
    weights_path = ROOT_DIR / "model1.h5"
    missing = [path.name for path in (architecture_path, weights_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The model files are missing: " + ", ".join(missing)
        )

    import keras
    from keras.models import Sequential, model_from_json

    with architecture_path.open("r", encoding="utf-8") as model_file:
        architecture = json.load(model_file)
    is_legacy_sequential = (
        architecture.get("class_name") == "Sequential"
        and isinstance(architecture.get("config"), list)
    )
    architecture = _normalize_legacy_sequential_config(architecture)
    if is_legacy_sequential:
        model = Sequential(name=architecture["config"]["name"])
        for layer_config in architecture["config"]["layers"]:
            layer_class = getattr(keras.layers, layer_config["class_name"], None)
            if layer_class is None:
                raise ValueError(
                    "Unsupported model layer: " + layer_config["class_name"]
                )
            model.add(layer_class.from_config(layer_config["config"]))
    else:
        model = model_from_json(json.dumps(architecture))
    model.load_weights(str(weights_path))
    MODEL = model
    return MODEL


def _normalize_legacy_sequential_config(architecture):
    """Adapt Keras 2.2 Sequential JSON to the current Keras config shape."""
    config = architecture.get("config")
    if architecture.get("class_name") != "Sequential" or not isinstance(config, list):
        return architecture

    layers = config
    if layers:
        first_layer_config = layers[0].get("config", {})
        input_shape = first_layer_config.pop("batch_input_shape", None)
        if input_shape is not None:
            input_layer = {
                "class_name": "InputLayer",
                "config": {
                    "batch_shape": input_shape,
                    "dtype": first_layer_config.get("dtype", "float32"),
                    "sparse": False,
                    "ragged": False,
                    "name": "input_layer",
                },
            }
            layers.insert(0, input_layer)

    architecture["config"] = {
        "name": "plant_leaf_cnn",
        "layers": layers,
    }
    return architecture


def display_label(label):
    """Turn a PlantVillage class name into a readable crop or condition."""
    return label.replace("_", " ").strip().title()


def prepare_image(image):
    """Create the CNN's 128×128 RGB batch without scaling pixel values."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a Pillow image.")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = image.convert("RGB").resize((128, 128), resampling)
    return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)


def predict_image(image, model=None):
    """Return the three highest-scoring model classes for a Pillow image."""
    classifier = model if model is not None else load_model()
    scores = np.asarray(classifier.predict(prepare_image(image), verbose=0)).reshape(-1)
    if scores.size != len(LABELS):
        raise ValueError(
            "The model returned {} classes; {} labels are configured.".format(
                scores.size, len(LABELS)
            )
        )
    if not np.isfinite(scores).all():
        raise ValueError("The model returned invalid prediction scores.")

    order = np.argsort(scores)[::-1][:3]
    return [(LABELS[int(index)], float(scores[index])) for index in order]
