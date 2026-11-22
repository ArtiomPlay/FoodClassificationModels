import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from dataset import load_dataset, CLASSES

# Configs
IMAGE_SIZE = (50, 50)
BATCH_SIZE = 32
EPOCHS = 30
RANDOM_STATE = 42
DATA_DIRECTORY = f"Processed/Size{IMAGE_SIZE[0]}"

# 1. Load dataset
print("Loading training dataset...")
train_images, train_labels = load_dataset(
    f"{DATA_DIRECTORY}/Train",
    image_size=IMAGE_SIZE,
    grayscale=True
)

print("Loading testing dataset...")
test_images, test_labels = load_dataset(
    f"{DATA_DIRECTORY}/Test",
    image_size=IMAGE_SIZE,
    grayscale=True
)

# 2. Prepare image data
X = train_images[..., np.newaxis].astype("float32") / 255.0
X_test = test_images[..., np.newaxis].astype("float32") / 255.0

y = train_labels
y_test = test_labels

print("\nDataset shapes:")
print("Training images: ", X.shape)
print("Training labels: ", y.shape)
print("Testing images: ", X_test.shape)
print("Testing labels: ", y_test.shape)

# 3. Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples: ", len(X_train))
print("Validation samples: ", len(X_val))
print("Testing samples: ", len(X_test))

# 4. Create FNN model
model = keras.Sequential([
    layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),

    layers.Dense(len(CLASSES), activation="softmax")
])

# 5. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

# 6. Train model
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nStarting training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

USED_EPOCHS = len(history.history["loss"])

# 7. Evaluate model on test dataset
print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

EXPERIMENT_NAME = f"FNN_B{BATCH_SIZE}_E{EPOCHS}_Used{USED_EPOCHS}_S{IMAGE_SIZE[0]}_Acc{test_accuracy * 100:.2f}%"

# 8. Generate predictions
predictions = model.predict(X_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

# 9. Classification report
report = classification_report(
    y_test,
    predicted_labels,
    labels=np.arange(len(CLASSES)),
    target_names=CLASSES,
    zero_division=0
)

print("\nClassification report:")
print(report)

with open(
    os.path.join("Reports", EXPERIMENT_NAME + "_Report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# 10. Confusion matrix
os.makedirs("Plots", exist_ok=True)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predicted_labels,
    labels=np.arange(len(CLASSES)),
    display_labels=CLASSES,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("FNN Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join("Plots", EXPERIMENT_NAME + "_ConfusionMatrix.png"), dpi=200)
plt.show()

# 11. Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(history.history["accuracy"], label="Training")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

# Loss
axes[1].plot(history.history["loss"], label="Training")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join("Plots", EXPERIMENT_NAME + "_TrainingHistory.png"), dpi=200)
plt.show()

# 12. Save trained model
os.makedirs("Models", exist_ok=True)
model_path = os.path.join(
    "Models",
    EXPERIMENT_NAME + ".keras"
)

model.save(model_path)

print(f"\nModel saved to: {model_path}")