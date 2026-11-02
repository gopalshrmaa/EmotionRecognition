
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Configuration
DATASET_DIR = r"C:\Users\DELL\Downloads\archive"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")

IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 15

# 2. Load training and validation images
print("Loading dataset...")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
print("Emotion classes:", class_names)

# 3. Optimize dataset loading
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# 4. Build CNN architecture
model = keras.Sequential([
    layers.Input(shape=(48, 48, 1)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.30),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.40),

    layers.Dense(7, activation="softmax")
])

# 5. Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Create model directory
os.makedirs("models", exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "models/best_emotion_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# 7. Train model
print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 8. Save trained model
model.save("models/emotion_model.keras")

print("\nTraining completed successfully!")
print("Model saved in models/emotion_model.keras")
