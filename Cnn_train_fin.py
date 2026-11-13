"""
Project: Plant Leaf Disease Detection using CNN
File: Cnn_train_fin.py

Purpose:
--------
This script trains a Convolutional Neural Network (CNN)
to detect plant leaf diseases using image classification.

Dataset Structure:
------------------
dataset/
    train/
        class1/
        class2/
        ...
    test/
        class1/
        class2/
        ...
"""

# ===============================
# Import Required Libraries
# ===============================

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator


# ===============================
# Step 1: Initialize CNN Model
# ===============================

model = Sequential()


# ===============================
# Step 2: Convolution + Pooling Layers
# ===============================

# Layer 1
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

# Layer 2
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

# Layer 3
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

# Layer 4
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

# Layer 5
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())


# ===============================
# Step 3: Flatten Layer
# ===============================

model.add(Dropout(0.2))
model.add(Flatten())


# ===============================
# Step 4: Fully Connected Layers
# ===============================

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))


# Output Layer (25 disease classes)
model.add(Dense(25, activation='softmax'))


# ===============================
# Step 5: Compile CNN
# ===============================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ===============================
# Step 6: Data Preprocessing
# ===============================

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=None,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescaling for testing
test_datagen = ImageDataGenerator(rescale=1./255)


# ===============================
# Step 7: Load Training Dataset
# ===============================

training_set = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

print("Training Labels:")
print(training_set.class_indices)


# ===============================
# Step 8: Load Testing Dataset
# ===============================

test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

print("Testing Labels:")
print(test_set.class_indices)


# ===============================
# Step 9: Train the CNN Model
# ===============================

model.fit_generator(
    training_set,
    steps_per_epoch=375,
    epochs=10,
    validation_data=test_set,
    validation_steps=125
)


# ===============================
# Step 10: Save the Trained Model
# ===============================

model_json = model.to_json()

with open("model1.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save_weights("model1.h5")

print("✅ Model saved to disk")