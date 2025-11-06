import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

@st.cache_resource
def load_emotion_model():
    return load_model("emotion_cnn.h5")

model = load_emotion_model()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.set_page_config(page_title="FACEMOJI", layout="centered")
st.title("FaceMoji")
st.write("Detects your facial emotion live and displays the matching emoji!")

col1, col2 = st.columns(2)
start_btn = col1.button("▶️ Start Mirror")
stop_btn = col2.button("⏹️ Stop Mirror")

FRAME_WINDOW = st.image([], channels="RGB")
status_text = st.empty()    

if "run" not in st.session_state:
    st.session_state.run = False

if start_btn:
    st.session_state.run = True
    status_text.info("Camera is running...")

if stop_btn:
    st.session_state.run = False
    status_text.warning("Camera stopped.")

cap = cv2.VideoCapture(0)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emojis = {label: cv2.imread(os.path.join("emojis", f"{label}.webp"), cv2.IMREAD_UNCHANGED)
          for label in emotion_labels}

def overlay_emoji(frame, emoji, x, y, w, h):
    if emoji is None:
        return frame
    emoji_resized = cv2.resize(emoji, (w, h))
    if emoji_resized.shape[2] == 4:
        alpha_s = emoji_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (alpha_s * emoji_resized[:, :, c] + alpha_l * frame[y:y+h, x:x+w, c])
    else:
        frame[y:y+h, x:x+w] = emoji_resized
    return frame


while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.error("Unable to access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
        roi = np.expand_dims(roi_gray, axis=(0, -1))

        pred = model.predict(roi)
        emotion = emotion_labels[np.argmax(pred)]

        emoji_img = emojis.get(emotion)
        frame = overlay_emoji(frame, emoji_img, x, y, w, h)
        cv2.putText(frame, emotion.upper(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
FRAME_WINDOW.image([])