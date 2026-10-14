# Real-Time Webcam Emotion Recognition Pipeline 🎥🧠

A real-time computer vision application that captures webcam video, detects faces, and classifies facial emotions using OpenCV, ONNX Runtime, and Streamlit.

This project was built as an educational pipeline to explore computer vision fundamentals—from raw matrix operations and bounding box spatial transformations to neural network inference and probability distributions.

## 🌟 Key Features

* **Low-Latency Streaming**: Browser-based video streaming powered by WebRTC via `streamlit-webrtc`.

* **Face Localization**: Real-time multi-scale face detection using Viola-Jones Haar Cascades.

* **Deep Learning Inference**: 8-class facial emotion classification using an optimized **FERPlus** convolutional neural network via **ONNX Runtime** (CPU-optimized, no bulky PyTorch required).

* **Computer Vision Fundamentals**: Explicit frame manipulation, including BGR-to-grayscale transformation, Region of Interest (ROI) slicing, contrast histogram equalization, and tensor reshaping ($NCHW$).

* **Modern Python Tooling**: Managed entirely with **`uv`** for reproducible environments and rapid dependency resolution.

## 📊 Pipeline Architecture

```
[Webcam Stream] 
       │ (BGR 640x480)
       ▼
[Grayscale Conversion] ──► [Haar Cascade Detection]
                                  │
                                  ▼
                        [Face Bounding Box (ROI)]
                                  │
                                  ▼
                     [Contrast Equalization (HE)]
                                  │
                                  ▼
                       [Resize & Tensor Formatting]
                           (1, 1, 64, 64) float32
                                  │
                                  ▼
                       [ONNX Model Forward Pass]
                                  │
                                  ▼
                        [Softmax Normalization]
                                  │
                                  ▼
                      [Annotated WebRTC Frame]

```

### Detected Emotions (FERPlus 8 Classes)

* **Neutral**

* **Happiness**

* **Surprise**

* **Sadness**

* **Anger**

* **Disgust**

* **Fear**

* **Contempt**

## 🚀 Getting Started

### Prerequisites

* **Python**: `>= 3.10` (Python 3.10 – 3.12 recommended for broadest CV ecosystem compatibility).

* **uv**: Modern, ultra-fast Python package and project manager.

  Install `uv` (if not already installed):

  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  
  ```

### Installation

1. **Clone the repository:**

   ```
   git clone <your-repo-url>
   cd emotions-cv
   
   ```

2. **Initialize and sync the virtual environment using `uv`:**

   ```
   uv venv
   source .venv/bin/activate
   uv pip install streamlit streamlit-webrtc opencv-python onnxruntime numpy av
   
   ```

3. **Download Required Model & Cascade Weights:**
   Download the Haar Cascade face detector and the FERPlus ONNX model into the project root:

   ```
   # Download Haar Cascade XML for face localization
   wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
   
   # Download FERPlus ONNX emotion model (~3 MB)
   wget -O emotion_model.onnx https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
   
   ```

## 💻 Running the App

Start the Streamlit application using `uv run`:

```
uv run streamlit run app.py

```

Open your browser at `http://localhost:8501`, grant webcam permissions, and click **START**.

## 📁 Project Structure

```
.
├── app.py                                # Main Streamlit app & WebRTC callback loop
├── emotion_model.onnx                    # Pretrained FERPlus ONNX model weights
├── haarcascade_frontalface_default.xml   # Pretrained Haar Cascade detector
├── pyproject.toml                        # Project metadata & dependencies
├── README.md                             # Documentation
└── .venv/                                # Managed virtual environment

```

## 🛠️ Concepts Explored in this Project

1. **Spatial Transformations & ROI Slicing**:
   Mapping dynamic, continuous pixel coordinates $(x, y, w, h)$ from a high-resolution canvas into discrete cropped sub-matrices.

2. **Contrast Normalization**:
   Applying `cv2.equalizeHist()` to counteract ambient lighting shifts, exposing subtle micro-features (e.g., eye widening, nose wrinkles) critical for detecting underrepresented emotion classes.

3. **Tensor Reshaping (**$NCHW$**)**:
   Transforming a 2D grayscale matrix `(H, W)` into a 4-dimensional inference tensor `[Batch, Channel, Height, Width]` required by convolutional neural networks.

4. **Logits vs. Softmax**:
   Converting raw, unbounded model activation scores into a normalized probability distribution using numerically stable softmax:
   

   $$
   \text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum e^{z_j - \max(z)}}
   $$

## 🔧 Troubleshooting & Known Environments

### Running Inside a Virtual Machine (Ubuntu on VirtualBox / VMware / WSL2)

* **USB Webcam Not Found (`/dev/video*` missing)**:
  Ensure USB pass-through is active on your host machine.

  * In **VirtualBox**: Ensure the VM's USB Controller is set to **USB 3.0 (xHCI)** (requires the VirtualBox Extension Pack).

  * In **VMware**: Go to `VM` → `Removable Devices` → `[Your Webcam]` → `Connect (Disconnect from host)`.

* **WebRTC Endless Loading / Spinning in Brave Browser**:

  * Open `brave://settings/privacy` and set **WebRTC IP handling policy** to **Default**.

  * Disable Brave Shields for `localhost`.

  * Alternatively, test using Firefox or Chromium.

* **Linux Device Permissions**:
  Add your current user to the `video` group:

  ```
  sudo usermod -aG video $USER
  newgrp video
  
  ```

## 📜 License

This project is licensed under the MIT License.