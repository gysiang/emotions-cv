import os
import av
import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from streamlit_webrtc import RTCConfiguration, webrtc_streamer

st.set_page_config(page_title="Real-Time Emotion Recognition", layout="wide")

st.title("Webcam Emotion Recognition Pipeline")
st.caption(
    "Pipeline: Haar Face Detection → Crop ROI → 64x64 Resizing → ONNX Inference → Softmax Probabilities"
)

# ----------------- Check & Load Assets -----------------
cascade_file = "haarcascade_frontalface_default.xml"
model_file = "emotion_model.onnx"

if not os.path.exists(cascade_file):
    st.error(
        f"Missing `{cascade_file}`. Run: `wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{cascade_file}`"
    )
    st.stop()

if not os.path.exists(model_file):
    st.error(
        f"Missing `{model_file}`. Run: `wget -O emotion_model.onnx https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx`"
    )
    st.stop()

# 1. Initialize Face Detector
face_cascade = cv2.CascadeClassifier(cascade_file)

# 2. Initialize ONNX Inference Session
ort_session = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
input_name = ort_session.get_inputs()[0].name

# FERPlus 8 emotion classes
EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def softmax(x: np.ndarray) -> np.ndarray:
    """Computes stable softmax probabilities from raw logits."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# ----------------- Frame Processing Callback -----------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    for x, y, w, h in faces:
        # 1. Slice Face Region of Interest (ROI)
        face_roi = gray[y : y + h, x : x + w]

        try:
            # 2. CV Preprocessing: Resize to 64x64 and cast to float32
            resized = cv2.resize(face_roi, (64, 64), interpolation=cv2.INTER_AREA)
            resized = resized.astype(np.float32)

            # 3. Reshape from (64, 64) -> (1, 1, 64, 64) [Batch, Channel, Height, Width]
            input_tensor = np.expand_dims(np.expand_dims(resized, axis=0), axis=0)

            # 4. ONNX Model Inference
            raw_logits = ort_session.run(None, {input_name: input_tensor})[0]

            # 5. Calculate probabilities & extract top prediction
            probs = softmax(raw_logits)[0]
            top_idx = int(np.argmax(probs))
            top_emotion = EMOTION_LABELS[top_idx]
            confidence = probs[top_idx] * 100

            display_text = f"{top_emotion.title()} ({confidence:.0f}%)"
            box_color = (
                (0, 255, 0)
                if top_emotion in ["happiness", "surprise"]
                else (0, 165, 255)
            )

        except Exception as e:
            display_text = "Face"
            box_color = (0, 255, 0)

        # 6. Render Visual Annotations onto the frame
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
        cv2.rectangle(img, (x, y - 25), (x + w, y), box_color, -1)
        cv2.putText(
            img,
            display_text,
            (x + 5, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------- UI Layout (Option A) -----------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### Live Feed")
    webrtc_streamer(
        key="emotion-detection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=True,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
    )

with col3:
    st.markdown("### CV Pipeline Debug")
    st.info("""
        **1. Detection:** Haar Cascades\n
        **2. ROI Alignment:** $(x, y, w, h)$ slice\n
        **3. Preprocessing:** $64\\times64$ Grayscale, `NCHW` tensor\n
        **4. Model:** FERPlus ONNX\n
        **5. Output:** 8-class Softmax Distribution\n
        """)
