
import cv2
import numpy as np
import tensorflow as tf

# =====================================
# 1. LOAD MODEL
# =====================================

MODEL_PATH = "models/best_emotion_model.keras"

print("Loading AI Emotion Model...")

model = tf.keras.models.load_model(MODEL_PATH)

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# =====================================
# 2. FACE DETECTOR
# =====================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError("Face detector not loaded!")

# =====================================
# 3. CAMERA SETUP
# =====================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("Camera not available!")

cv2.namedWindow(
    "AI Emotion Recognition Dashboard",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "AI Emotion Recognition Dashboard",
    1100, 650
)

# =====================================
# 4. PREDICTION SMOOTHING
# =====================================

smooth_predictions = None
SMOOTHING = 0.25

print("AI Dashboard Started!")
print("Press Q to quit.")

# =====================================
# 5. MAIN LOOP
# =====================================

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70)
    )

    # Select largest face
    faces = sorted(
        faces,
        key=lambda item: item[2] * item[3],
        reverse=True
    )

    current_predictions = None
    current_emotion = "No face detected"
    confidence = 0.0

    if len(faces) > 0:

        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (48, 48)
        )

        face = face.astype("float32")

        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # CNN prediction
        predictions = model.predict(
            face,
            verbose=0
        )[0]

        # Smooth predictions
        if smooth_predictions is None:
            smooth_predictions = predictions

        else:
            smooth_predictions = (
                SMOOTHING * predictions +
                (1 - SMOOTHING) * smooth_predictions
            )

        current_predictions = smooth_predictions

        emotion_index = np.argmax(
            current_predictions
        )

        current_emotion = EMOTIONS[emotion_index]

        confidence = (
            current_predictions[emotion_index] * 100
        )

        # Draw face bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 220, 120),
            2,
            cv2.LINE_AA
        )

        # Emotion label
        label = f"{current_emotion} | {confidence:.1f}%"

        cv2.rectangle(
            frame,
            (x, max(0, y-35)),
            (x+w, y),
            (0, 220, 120),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x+5, max(23, y-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (15, 20, 25),
            2,
            cv2.LINE_AA
        )

    else:
        smooth_predictions = None

    # =================================
    # 6. CREATE DASHBOARD
    # =================================

    height, width = frame.shape[:2]

    panel_width = 330
    header_height = 60

    dashboard = np.zeros(
        (height + header_height, width + panel_width, 3),
        dtype=np.uint8
    )

    dashboard[:] = (24, 29, 40)

    # Header
    cv2.rectangle(
        dashboard,
        (0, 0),
        (width + panel_width, header_height),
        (32, 40, 54),
        -1
    )

    cv2.putText(
        dashboard,
        "AI FACIAL EMOTION RECOGNITION",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Camera feed
    dashboard[
        header_height:header_height+height,
        0:width
    ] = frame

    # Right side panel
    px = width

    cv2.rectangle(
        dashboard,
        (px, header_height),
        (width+panel_width, height+header_height),
        (28, 34, 48),
        -1
    )

    cv2.putText(
        dashboard,
        "LIVE ANALYTICS",
        (px+20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 180),
        2
    )

    cv2.putText(
        dashboard,
        "DETECTED EMOTION",
        (px+20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 190, 205),
        1
    )

    cv2.putText(
        dashboard,
        current_emotion.upper(),
        (px+20, 177),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 230, 150),
        2
    )

    cv2.putText(
        dashboard,
        f"Model score: {confidence:.1f}%",
        (px+20, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 225, 235),
        1
    )

    cv2.putText(
        dashboard,
        "EMOTION SCORES",
        (px+20, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # Seven emotion probability bars
    if current_predictions is not None:

        for i, emotion in enumerate(EMOTIONS):

            score = float(current_predictions[i]) * 100

            bar_y = 280 + i * 34

            cv2.putText(
                dashboard,
                emotion,
                (px+20, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (220, 225, 235),
                1
            )

            cv2.putText(
                dashboard,
                f"{score:.1f}%",
                (px+255, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1
            )

            # Background bar
            cv2.rectangle(
                dashboard,
                (px+20, bar_y+7),
                (px+300, bar_y+15),
                (65, 72, 88),
                -1
            )

            # Prediction bar
            bar_length = int(
                280 * np.clip(score / 100, 0, 1)
            )

            cv2.rectangle(
                dashboard,
                (px+20, bar_y+7),
                (px+20+bar_length, bar_y+15),
                (0, 210, 145),
                -1
            )

    else:

        cv2.putText(
            dashboard,
            "Waiting for face...",
            (px+20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 190, 205),
            1
        )

    # Footer
    cv2.putText(
        dashboard,
        "Q - EXIT",
        (px+20, height+35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 220, 180),
        1
    )

    cv2.imshow(
        "AI Emotion Recognition Dashboard",
        dashboard
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================================
# 7. CLEANUP
# =====================================

camera.release()
cv2.destroyAllWindows()

print("Dashboard closed successfully!")
