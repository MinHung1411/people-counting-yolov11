<div align="center">

# 🚶‍♂️ People Counting in Public Spaces
### Real-time Person Detection & Counting using YOLOv11 + BoT-SORT

[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20LIVE%20DEMO-Try%20Now!-FF6B35?style=for-the-badge)](https://minhunhooo-crowd-people-counter.hf.space/)&nbsp;&nbsp;[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://minhunhooo-crowd-people-counter.hf.space/)&nbsp;&nbsp;[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com)

**🔗 Live Demo: [Crowd People Counter Web App](https://minhunhooo-crowd-people-counter.hf.space/)**

---

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-FF6B35?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.9-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Training Pipeline](#-training-pipeline)
- [Inference](#-inference)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Technical Details](#-technical-details)
- [Author](#-author)
- [References](#-references)

---

## 🔍 Overview

This project implements a **real-time people counting system** for public spaces using state-of-the-art computer vision techniques. The system is capable of:

- 🖼️ **Detecting and counting people in static images** with geometric filtering to reduce false positives
- 🎬 **Tracking and counting people in real-time video streams** using BoT-SORT multi-object tracker
- 🌐 **Live web application** deployed on Hugging Face Spaces with Gradio UI
- 📦 **Exporting the trained model** to ONNX and TorchScript for production deployment

The model was **fine-tuned from YOLOv11m** (medium) on the CrowdHuman dataset — one of the most challenging crowd detection benchmarks available — achieving strong performance even in dense crowd scenarios.

> 🎓 **Academic Context:** End-of-semester project for the course *Computer Vision and Applications* (Course code: 420301436401), Faculty of Computer Science, Academic Year 2025–2026, Semester I.

---

## 🌐 Live Demo

The application is deployed on **Hugging Face Spaces** and can be accessed directly in the browser:

🔗 **https://minhunhooo-crowd-people-counter.hf.space/**

| Feature | Description |
|---------|-------------|
| 📷 **Image Detection** | Upload any image → instant person count with annotated bounding boxes |
| 🎬 **Video Tracking** | Upload a video → processed output with ByteTrack multi-object tracking |
| 🎥 **Live Camera** | Real-time webcam detection with adjustable confidence thresholds |

**Configurable Parameters:**
- Confidence threshold (0.1 – 0.9)
- IoU threshold for NMS
- Processing resolution (320px – 960px)
- Frame skip for video processing
- Maximum processing time limit

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Model** | YOLOv11m (fine-tuned) |
| **Dataset** | CrowdHuman |
| **mAP@50** | **95.8%** |
| **mAP@50-95** | **84.8%** |
| **Precision** | **88.0%** |
| **Recall** | **90.1%** |
| **Training Epochs** | 50 |
| **Image Size** | 640×640 |
| **Inference Speed** | ~8ms per frame (Tesla T4) |
| **Val Images** | 1,169 |
| **Val Detections** | 6,037 |

> 📈 Results evaluated on CrowdHuman validation set after 50 epochs of training on Kaggle (NVIDIA Tesla T4 GPU).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT PIPELINE                        │
│         Image / Video Frame (640×640 resized)           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  YOLOv11m BACKBONE                       │
│     CSP + C3k2 + SPPF + Attention Modules               │
│     Pretrained on COCO → Fine-tuned on CrowdHuman       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  DETECTION HEAD                          │
│     Multi-scale predictions: P3 / P4 / P5              │
│     Single class: "person" (nc=1)                       │
│     NMS (IoU threshold: 0.5)                            │
└─────────────────────────┬───────────────────────────────┘
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│   IMAGE MODE         │  │   VIDEO MODE                 │
│                      │  │                              │
│  Geometric Filter:   │  │  BoT-SORT Tracker:           │
│  • height ≥ 90px     │  │  • ID-consistent tracking    │
│  • width ≥ 40px      │  │  • conf ≥ 0.5                │
│  • aspect ratio ≤ 6  │  │  • height ≥ 120px            │
│  • area ≥ 9000px²    │  │  • width ≥ 60px              │
└──────────┬───────────┘  └──────────────┬───────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────────────────────────────────────────┐
│                    OUTPUT                                 │
│   Annotated image/video + People count overlay           │
└──────────────────────────────────────────────────────────┘
```

---

## 📦 Dataset

**[CrowdHuman](https://www.crowdhuman.org/)** — A benchmark dataset specifically designed for detecting people in crowded scenes.

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | ~15,000 | ~340,000 persons |
| Val   | ~4,370  | ~99,000 persons |

**Key characteristics:**
- Images contain **highly overlapping** and **occluded** pedestrians
- Average **~23 persons per image** (far denser than COCO)
- Sourced from: Kaggle dataset `menhari/crowd-human-crowd-detection`

**YAML Configuration:**
```yaml
train: /kaggle/working/CrowdHuman/train/images
val:   /kaggle/working/CrowdHuman/val/images
nc: 1
names: ['person']
```

---

## 🏋️ Training Pipeline

### Environment
- Platform: **Kaggle Notebooks** (free GPU tier)
- GPU: **NVIDIA Tesla T4** (15 GB VRAM)
- Framework: **Ultralytics 8.3.225**, **PyTorch 2.6.0+cu124**

### Augmentation Strategy
```python
model.train(
    epochs=50,
    imgsz=640,
    batch=16,
    # Geometric augmentations
    scale=0.9,
    translate=0.2,
    degrees=15,
    fliplr=0.5,
    # Advanced augmentations
    mosaic=1.0,        # Mosaic 4-image augmentation
    mixup=0.5,         # MixUp augmentation
    copy_paste=0.3,    # Copy-Paste (great for crowds!)
    # Anti-overfitting
    patience=15,       # Early stopping
    dropout=0.3,
    label_smoothing=0.1,
    weight_decay=0.0005,
    close_mosaic=10,   # Disable mosaic last 10 epochs
    warmup_epochs=3,
)
```

### Training Stages
1. **Warmup** (epochs 1–3): Gradual learning rate increase
2. **Main Training** (epochs 4–40): Full augmentation suite active
3. **Stabilization** (epochs 41–50): Mosaic disabled for cleaner convergence
4. **Early stopping**: Monitors val mAP, stops if no improvement for 15 consecutive epochs

---

## 🔎 Inference

### Image Inference
```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
img = cv2.imread("crowd.jpg")

results = model(img, conf=0.35, iou=0.5, classes=0)[0]

count = 0
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    height, width = y2 - y1, x2 - x1
    
    # Geometric filter: remove false positives
    if height < 90 or width < 40 or height/width > 6 or height*width < 9000:
        continue
    count += 1
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

print(f"Detected {count} people")
```

### Video Inference with BoT-SORT Tracking
```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.5,
        verbose=False
    )[0]

    current_people = sum(
        1 for box in results.boxes
        if (box.xyxy[0][3] - box.xyxy[0][1]) >= 120
        and (box.xyxy[0][2] - box.xyxy[0][0]) >= 60
    )
    
    cv2.putText(frame, f"CURRENT PEOPLE: {current_people}",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 4)
    
    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Model Export
```python
model_export = YOLO("best.pt")

# Export to ONNX (for OpenCV DNN, TensorRT, cross-platform)
model_export.export(format="onnx", imgsz=1280)

# Export to TorchScript (for C++ / mobile deployment)
model_export.export(format="torchscript", imgsz=1280)
```

---

## 🚀 Deployment

The application is deployed as a **Gradio web app** on [Hugging Face Spaces](https://huggingface.co/spaces).

### Deployment Stack

| Component | Technology |
|-----------|------------|
| **Web Framework** | Gradio 5.9 |
| **ML Runtime** | Ultralytics (YOLOv11) |
| **Tracker** | ByteTrack (optimized for web) |
| **Platform** | Hugging Face Spaces (CPU) |
| **CI/CD** | Git-based auto-deploy |

### Features of the Web App
- **📷 Image Detection** — Upload any image, adjust confidence threshold, get annotated results
- **🎬 Video Processing** — Upload video with configurable frame skip, IoU, resolution, and time limit
- **🎥 Live Camera** — Real-time webcam detection with streaming output
- **⚙️ Tunable Parameters** — Confidence, IoU, resolution, min-hits for tracking stability

### Local Development
```bash
cd deployment
pip install -r requirements.txt
python app.py
```

> **Note:** Requires `best.pt` (trained model weights) in the same directory as `app.py`.

### Deployment Files

```
deployment/
├── app.py              # Gradio application (image, video, webcam tabs)
├── requirements.txt    # Python dependencies
└── README.md           # Hugging Face Spaces metadata
```

---

## 📁 Project Structure

```
people-counting-yolov11/
│
├── 📓 notebook/
│   └── people_counting_yolov11.ipynb   # Full training & inference pipeline
│
├── 🚀 deployment/
│   ├── app.py                           # Gradio web application
│   ├── requirements.txt                 # Deployment dependencies
│   └── README.md                        # HF Spaces config
│
├── 🖼️ assets/
│   └── banner.png                       # Project banner
│
├── 📄 README.md                         # Project documentation (this file)
├── 📄 LICENSE                           # MIT License
└── 📄 .gitignore
```

**Notebook sections:**
| Section | Description |
|---------|-------------|
| Cell 1 | Project info & team details |
| Cell 2-3 | Dataset download (Kaggle Hub) & file listing |
| Cell 4 | Install dependencies (`ultralytics`, `opencv-python`) |
| Cell 5 | Dataset preparation & YAML config creation |
| Cell 6 | **YOLOv11m training** (50 epochs, full augmentation) |
| Cell 7 | Model export (ONNX + TorchScript) |
| Cell 8 | Download trained model weights |
| Cell 9-13 | **Image inference** with geometric filtering |
| Cell 14-19 | **Video inference** with BoT-SORT tracking |

---

## ⚡ Quick Start

### 1. Try the Live Demo (No Setup Required)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20Live%20Demo-Hugging%20Face-blue?style=for-the-badge)](https://minhunhooo-crowd-people-counter.hf.space/)

### 2. Open the Notebook on Kaggle / Google Colab

Click to open in Kaggle:

[![Kaggle](https://img.shields.io/badge/Run%20on-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com)

Or open in Google Colab:

[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

### 3. Install Dependencies

```bash
pip install ultralytics==8.3.225 opencv-python matplotlib gradio
```

### 4. Run Training (requires GPU)

Open `notebook/people_counting_yolov11.ipynb` and run all cells sequentially.

### 5. Inference on Your Own Images/Videos

After training is complete:
- **Image:** Upload any image → get annotated result with person count
- **Video:** Upload any MP4 → get processed video with real-time counter
- **Web App:** Launch the Gradio app for an interactive demo

---

## 🔧 Technical Details

### Why YOLOv11m?
- **YOLOv11m** (medium variant) offers an excellent trade-off between accuracy and speed
- Significantly more accurate than YOLOv8 in dense crowd scenarios due to improved attention mechanisms
- Pretrained on **COCO** (which includes a `person` class), enabling effective fine-tuning with fewer epochs

### Why BoT-SORT for Video?
- **BoT-SORT** (Boosted Object Tracker with Sort) combines Kalman filtering + appearance features
- Maintains stable **track IDs** across frames even with occlusion
- Prevents the same person from being counted multiple times when they momentarily overlap

### Geometric Filtering Strategy
To reduce false positives (shadows, small objects, detection errors):

| Filter | Image Mode | Video Mode | Purpose |
|--------|-----------|-----------|---------| 
| Min height | ≥ 90px | ≥ 120px | Remove tiny detections |
| Min width | ≥ 40px | ≥ 60px | Remove narrow artifacts |
| Max aspect ratio | ≤ 6 | — | Remove pole-like false positives |
| Min area | ≥ 9,000px² | — | Remove small false positives |

### Confidence Thresholds
| Mode | Threshold | Rationale |
|------|-----------|-----------| 
| Image | `conf=0.35` | Balanced: captures partially visible people |
| Video | `conf=0.50` | Stricter: reduces noisy inter-frame detections |

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/MinHung1411">
        <img src="https://github.com/MinHung1411.png" width="100px;" alt="Hồ Minh Hưng"/><br />
        <sub><b>Hồ Minh Hưng</b></sub>
      </a><br />
      <sub>MSSV: 22666391</sub>
    </td>
  </tr>
</table>

### Responsibilities

| Area | Details |
|------|---------|
| 🧠 **Model Training** | Fine-tuned YOLOv11m on CrowdHuman dataset (50 epochs) with advanced augmentation |
| 📊 **Dataset Preparation** | Downloaded, preprocessed, and configured CrowdHuman for YOLO format |
| 🎬 **Video Inference** | Implemented BoT-SORT tracking pipeline with geometric filtering |
| 🚀 **Deployment** | Built Gradio web app and deployed to Hugging Face Spaces |
| 📝 **Documentation** | Project README, architecture diagrams, and technical documentation |

### Project Context

- **Course:** Computer Vision and Applications (THỊ GIÁC MÁY TÍNH VÀ ỨNG DỤNG)
- **Class:** DHKHMT18A
- **Course Code:** 420301436401
- **Supervisor:** Dr. Lê Thị Vĩnh Thanh
- **Institution:** University – Faculty of Computer Science
- **Academic Year:** 2025–2026, Semester I

### Collaborator

| Name | Student ID | Role |
|------|-----------|------|
| **Huỳnh Nhật Hoàng** | 22667391 | Image inference, post-processing, evaluation & report |

---

## 📚 References

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com)
- [CrowdHuman Dataset](https://www.crowdhuman.org/)
- [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651)
- [Kaggle Dataset: crowd-human-crowd-detection](https://www.kaggle.com/datasets/menhari/crowd-human-crowd-detection)
- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

Made with ❤️ by **Hồ Minh Hưng** — DHKHMT18A

</div>
