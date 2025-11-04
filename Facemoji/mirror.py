import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_cnn.h5")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

emojis = {label: cv2.imread(f"emojis/{label}.webp", -1) for label in emotion_labels}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def overlay_emoji(frame, emoji, x, y, w, h):
    emoji_resized = cv2.resize(emoji, (w, h))

    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(center)
    cv2.circle(mask, center, radius, 255, -1)

    if emoji_resized.shape[2] == 4:
        alpha_s = (emoji_resized[:, :, 3] / 255.0) * (mask / 255.0)
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            frame[y:y+h, x:x+w, c] = (alpha_s * emoji_resized[:, :, c] + alpha_l * frame[y:y+h, x:x+w, c])
    else:
        for c in range(3):
            frame[y:y+h, x:x+w, c] = np.where(mask == 255, emoji_resized[:, :, c], frame[y:y+h, x:x+w, c])


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        emoji = emojis[emotion]
        emoji = cv2.resize(emoji, (w, h))

        if emoji.shape[2] == 4:  # RGBA
            alpha_s = emoji[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(3):
                frame[y:y+h, x:x+w, c] = (alpha_s * emoji[:, :, c] + alpha_l * frame[y:y+h, x:x+w, c])
        else:
            overlay_emoji(frame, emoji, x, y, w, h)

        cv2.putText(frame, emotion.upper(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("AI Mirror", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
