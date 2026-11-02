# 😊 AI-Based Real-Time Emotion Recognition System

An AI-powered facial emotion recognition system that detects and classifies human emotions from live webcam video using Deep Learning and Computer Vision.

The project uses a trained CNN-based model to analyze facial expressions and display predicted emotions through a real-time dashboard.

## 🚀 Features

- Real-time facial emotion detection using a webcam.
- Deep Learning-based emotion classification.
- Face detection and image preprocessing.
- CNN model training and evaluation.
- Model testing and performance evaluation.
- Interactive emotion recognition dashboard.
- Live prediction visualization.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| TensorFlow / Keras | Deep Learning model |
| OpenCV | Webcam and image processing |
| NumPy | Numerical operations |
| CNN | Facial emotion classification |

## 🧠 How It Works

1. **Webcam Input:** Captures live video frames.
2. **Face Detection:** Identifies faces in the video.
3. **Preprocessing:** Prepares facial images for model input.
4. **Emotion Prediction:** The trained CNN model predicts the emotion.
5. **Dashboard:** Displays the detected emotion in real time.

## 📂 Project Structure

```text
EmotionRecognition/
│
├── models/
│   ├── best_emotion_model.keras
│   └── emotion_model.keras
│
├── src/
│   ├── check_dataset.py
│   ├── evaluate.py
│   ├── test_model.py
│   ├── train.py
│   └── webcam.py
│
├── dataset/              # Local dataset (not uploaded)
├── results/              # Generated results
├── .gitignore
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/gopalshrmaa/EmotionRecognition.git
cd EmotionRecognition
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install tensorflow opencv-python numpy
```

Install any additional dependencies required by the project.

### 5. Run the project

```bash
python src/webcam.py
```

Allow webcam access when prompted.

Press **Q** to close the application.

## 📊 Model Training & Evaluation

Train the model:

```bash
python src/train.py
```

Evaluate model performance:

```bash
python src/evaluate.py
```

Test the trained model:

```bash
python src/test_model.py
```

Check the dataset:

```bash
python src/check_dataset.py
```

## 🎯 Project Objectives

- Explore Deep Learning applications in facial emotion recognition.
- Build a real-time computer vision application.
- Understand CNN-based image classification.
- Integrate a trained model with webcam-based predictions.

## 🔮 Future Improvements

- Improve emotion classification accuracy.
- Add more robust facial emotion categories.
- Enhance dashboard design and visualization.
- Improve performance under different lighting conditions.
- Explore advanced CNN architectures.

## 👨‍💻 Author

**Gopal Sharma**

B.Tech | Artificial Intelligence & Machine Learning

GitHub: [@gopalshrmaa](https://github.com/gopalshrmaa)

---

⭐ If you find this project interesting, consider giving it a star!

**Made with ❤️ using Python, Deep Learning & Computer Vision.**