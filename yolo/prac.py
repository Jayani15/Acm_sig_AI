import cv2
import streamlit as st
from ultralytics import YOLO

model = YOLO(r"D:\ACM_AI\DL\yolo\prac.py")
st.title("YOLO Object Detection with Webcam 🎥")

# Buttons for starting/stopping video
start_stream = st.button("Start Video")
stop_stream = st.button("Stop Video")

# Display placeholder for video frames
frame_placeholder = st.empty()

# Function to capture and display video with YOLO detection
def live_yolo_stream():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        st.error("Error: Could not open video capture")
        return

    cap.set(3, 640)  # Set frame width
    cap.set(4, 480)  # Set frame height

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        # Run YOLO inference
        results = model(frame, conf=0.8)

        # Get annotated frame
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

        # Check if "Stop Video" button is pressed
        if stop_stream:
            break

    cap.release()

# Run live video feed if button is clicked
if start_stream:
    live_yolo_stream()
