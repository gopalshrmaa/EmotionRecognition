
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 1. Paths
DATASET_DIR = r"C:\Users\DELL\Downloads\archive"
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_PATH = "models/best_emotion_model.keras"

# 2. Load best trained model
print("Loading trained model...")
model = keras.models.load_model(MODEL_PATH)

# 3. Load test images
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=False,
    label_mode="int"
)

class_names = test_ds.class_names
print("\nEmotion classes:", class_names)

# 4. Evaluate model
print("\nEvaluating model on unseen test images...")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTEST RESULTS")
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
print("Test Loss:", round(test_loss, 4))

# 5. Generate predictions
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])

predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# 6. Classification report
print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_true,
    y_pred,
    labels=range(len(class_names)),
    target_names=class_names,
    zero_division=0
))

# 7. Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", xticks_rotation=45)
plt.title("Facial Emotion Recognition - Confusion Matrix")
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix.png")
plt.show()

print("\nEvaluation completed!")
print("Confusion matrix saved in results/")
