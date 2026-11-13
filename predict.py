"""
Project: Plant Leaf Disease Detection using CNN
File: predict_fin.py

Purpose:
--------
This script loads a trained CNN model and predicts
the disease from a given plant leaf image.

Model Files Required:
---------------------
1. model1.json  -> Model Architecture
2. model1.h5    -> Model Weights
"""

# ===============================
# Import Required Libraries
# ===============================

import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json


# ===============================
# Load Trained Model
# ===============================

# Load model architecture
json_file = open("model1.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

# Load trained weights
loaded_model.load_weights("model1.h5")

print("✅ Model successfully loaded from disk")


# ===============================
# Class Labels (PlantVillage Dataset)
# ===============================

labels = [
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
    "Tomato___Tomato_mosaic_virus"
]


# ===============================
# Load Image for Prediction
# ===============================

# Change the path to test different images
test_image = image.load_img(
    "im_for_testing_purpose/p.healthy.JPG",
    target_size=(128, 128)
)

# Convert image to array
test_image = image.img_to_array(test_image)

# Expand dimensions to match model input
test_image = np.expand_dims(test_image, axis=0)


# ===============================
# Make Prediction
# ===============================

result = loaded_model.predict(test_image)

# Print raw prediction probabilities
print("Prediction probabilities:")
print(result)

# Get the predicted class
predicted_label = labels[result.argmax()]

print("\n🌿 Predicted Disease:")
print(predicted_label)