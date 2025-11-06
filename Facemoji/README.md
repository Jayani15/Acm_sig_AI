# 🪞 Facemoji – Real-Time Emotion-to-Emoji Translator

**FaceMoji** is an interactive computer vision project that detects your facial expressions in real-time using a **CNN emotion recognition model** and overlays matching **emoji reactions** directly onto your face using **OpenCV**.  
Think of it as your very own **AI-powered Snapchat mirror** 😎

---

## 🎬 Demo

> 🧠 Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise  
> 😄 Overlays expressive emojis live on your face feed  
> 📸 Built with OpenCV + TensorFlow + Custom CNN

---

## 🧠 Features

- 🎭 Real-time emotion recognition using a CNN trained on the **FER2013** dataset  
- 🟡 Emoji overlays that match facial emotions live  
- 👀 Face detection using **OpenCV Haar Cascades**  
- 💡 Circular emoji masking for clean overlay (only the round part covers the face)  
- 🧰 Modular code — plug in any model or emotion set you want  
- 💻 Works on webcam feeds or video files  

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| 🧠 Model | TensorFlow / Keras CNN |
| 📷 Vision | OpenCV |
| 🧩 Dataset | FER2013 (7 emotion classes) |
| 🖼️ Emojis | Transparent PNG images (48×48 / 128×128) |
| 💻 Language | Python 3.8+ |


---

## 🚀 Setup & Usage

### 1️⃣ Step 1: Download / Copy Project

Simply copy the **Face_emoji** folder to your local system.  
Make sure the following files are in the same directory:
- `app.py`
- `emotion_cnn.h5`
- `emojis/` folder containing your PNG emoji files.

---

### 2️⃣ Step 2: Install Dependencies

Open a terminal inside your project folder and run:

```bash
pip install tensorflow opencv-python streamlit numpy

If you prefer virtual environment:
python -m venv venv
venv\Scripts\activate  # (on Windows)
pip install -r requirements.txt

```
---
🪄 How It Works

Face Detection:
OpenCV locates faces in each webcam frame.

Emotion Recognition:
The cropped face is resized and passed to your trained CNN.

Emoji Overlay:
The detected emotion selects the corresponding PNG from /emojis/ and overlays it over your face with transparency masking.

Live Display:
Streamlit continuously renders frames in real time through your browser interface.

---

🧑‍💻 Author

Jayani Immidi
💡 AI Engineer • Computer Vision Enthusiast

📬 Made with ❤️ and OpenCV
