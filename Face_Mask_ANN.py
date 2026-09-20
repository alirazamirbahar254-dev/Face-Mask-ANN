# BINARY IMAGE CLASSIFICATION USING ANN
# Face Mask Dataset - With Mask / Without Mask

# =========================================================
# 1. Install Required Library
# =========================================================

!pip install split-folders -q


# =========================================================
# 2. Import Libraries
# =========================================================

import os
import splitfolders
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array


# =========================================================
# 3. Dataset Structure
# =========================================================
# facemask-dataset/
#     dataset/
#         with_mask/
#         without_mask/


# =========================================================
# 4. Split Dataset
# =========================================================

splitfolders.ratio(
    "facemask-dataset/dataset",
    output="output_dataset",
    seed=42,
    ratio=(0.7, 0.2, 0.1)
)

print("Dataset split complete!")


# =========================================================
# 5. Image Preprocessing and Normalization
# =========================================================

img_size = (64, 64)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)


# =========================================================
# 6. Load Training Data
# =========================================================

train_data = train_datagen.flow_from_directory(
    "output_dataset/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)


# =========================================================
# 7. Load Validation Data
# =========================================================

val_data = val_datagen.flow_from_directory(
    "output_dataset/val",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)


# =========================================================
# 8. Load Test Data
# =========================================================

test_data = test_datagen.flow_from_directory(
    "output_dataset/test",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

print("Class indices:", train_data.class_indices)


# =========================================================
# 9. Build ANN Model
# =========================================================

model = tf.keras.Sequential([
    
    tf.keras.Input(shape=(64, 64, 3)),
    
    # Convert image into 1D array
    tf.keras.layers.Flatten(),
    
    # Hidden Layer 1
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),
    
    # Dropout
    tf.keras.layers.Dropout(0.3),
    
    # Hidden Layer 2
    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),
    
    # Output Layer
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# =========================================================
# 10. Compile Model
# =========================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 11. Display Model Summary
# =========================================================

model.summary()


# =========================================================
# 12. Train ANN Model
# =========================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


# =========================================================
# 13. Evaluate Model on Test Data
# =========================================================

test_loss, test_accuracy = model.evaluate(test_data)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy Percentage:", test_accuracy * 100)


# =========================================================
# 14. Training and Validation Accuracy Graph
# =========================================================

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.show()


# =========================================================
# 15. Training and Validation Loss Graph
# =========================================================

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.show()


# =========================================================
# 16. Predict New Image
# =========================================================

# Upload a new image before running this section

from google.colab import files

uploaded = files.upload()

image_path = list(uploaded.keys())[0]


# =========================================================
# 17. Preprocess New Image
# =========================================================

img = load_img(
    image_path,
    target_size=(64, 64)
)

img_array = img_to_array(img) / 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
)


# =========================================================
# 18. Make Prediction
# =========================================================

prediction = model.predict(img_array)[0][0]

print("Prediction Score:", prediction)


# =========================================================
# 19. Display Prediction
# =========================================================

if prediction >= 0.5:
    print("Prediction: Without Mask")
else:
    print("Prediction: With Mask")


# =========================================================
# 20. Save Trained Model
# =========================================================

model.save("face_mask_ann.keras")

print("Model saved successfully!")
