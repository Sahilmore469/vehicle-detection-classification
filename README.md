<div align="center">

# 🚗 Vehicle Detection & Fine-Grained Classification

### A two-stage computer vision pipeline that finds vehicles and identifies their exact make, model, and year

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![ResNet50](https://img.shields.io/badge/Classifier-ResNet50-8A2BE2)]()
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://vehicle-detection-classification-mt3cj32atsaqx2qinw2mrn.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br>

### 🔗 [**Try the live demo →**](https://vehicle-detection-classification-mt3cj32atsaqx2qinw2mrn.streamlit.app)

</div>

---

## 📸 Demo

<div align="center">

*[Add a screenshot or GIF here of the app correctly identifying a car — this is the single most valuable thing you can put at the top of this README]*

</div>

---

## 🧠 What This Project Does

Most object detectors can tell you *"that's a car."* This pipeline goes a step further — it tells you **exactly which car**.

```
📷 Input Image
        │
        ▼
 ┌─────────────────┐        finds & localizes every vehicle
 │   YOLOv8         │───────────────────────────────────────┐
 │  (Detection)     │                                         │
 └─────────────────┘                                         ▼
                                                    ┌─────────────────┐
                                                    │   Crop & Resize  │
                                                    └─────────────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────┐        classifies exact
                                                    │   ResNet50       │───────  make / model / year
                                                    │ (Classification)│         (196 classes)
                                                    └─────────────────┘
                                                             │
                                                             ▼
                                            🏷️ "2012 BMW M3 Coupe" (91% confidence)
```

**Real-world analogues:** this two-stage detect-then-classify pattern is the same one used in insurance damage assessment, automated toll/parking systems, dealership inventory scanning, and traffic analytics platforms.

---

## ✨ Features

- 🔍 **Two-stage pipeline** — YOLOv8 detection feeding into a fine-tuned ResNet50 classifier
- 🎯 **196-class fine-grained classification** on the Stanford Cars dataset
- 📊 **Rigorous evaluation** — top-1/top-5 accuracy, precision, recall, F1 (macro + weighted)
- 🔬 **Progressive unfreezing** fine-tuning strategy for efficient transfer learning
- 🧊 **Grad-CAM interpretability** — visual confirmation the model attends to grilles/headlights/badges, not background noise
- 🌐 **Live, deployed demo** on Streamlit Community Cloud — no setup required to try it
- 📦 Clean, reproducible training pipeline in a single Colab notebook

---

## 🏗️ Architecture

| Stage | Model | Role |
|---|---|---|
| **1. Detection** | YOLOv8n (Ultralytics, pretrained on COCO) | Locates every vehicle in the frame and draws a bounding box |
| **2. Classification** | ResNet50 (fine-tuned) | Classifies each cropped vehicle into 1 of 196 make/model/year classes |

<details>
<summary><strong>📐 Why two stages instead of one?</strong></summary>
<br>
A single end-to-end model would have to learn localization and fine-grained recognition simultaneously, which is harder to train and debug. Splitting the problem lets each model specialize: YOLO is fast and reliable at "where is the vehicle," while ResNet50 gets a clean, cropped, standardized view to focus purely on fine-grained visual differences (grille shape, badge placement, headlight design) between visually similar classes.
</details>

---

## 📊 Results

| Metric | Score |
|---|---|
| **Top-1 Accuracy** | **83.86%** |
| **Top-5 Accuracy** | **96.84%** |
| Macro F1 | 0.83+ |
| Weighted F1 | 0.86 |

<details>
<summary><strong>📉 Where the model struggles most</strong></summary>
<br>

Analyzing per-class F1 scores surfaced a genuinely useful pattern: the lowest-performing classes weren't random — they clustered around **visually similar body styles across brands**. For example, "Chevrolet Express Van 2007" was most frequently confused with other cargo vans (Chevrolet Express Cargo Van, GMC Savana Van) rather than unrelated vehicle types. This confirms the model is learning genuine visual structure rather than memorizing noise.

</details>

<details>
<summary><strong>🔥 Grad-CAM interpretability check</strong></summary>
<br>

Grad-CAM visualizations on the classifier's final convolutional layer show activation concentrated on the **front grille, headlights, and wheel arches** — the same features a human would use to distinguish similar sedans. This is a meaningful sanity check: the model isn't keying on background artifacts or watermarks.

</details>

---

## 🗂️ Dataset

**[Stanford Cars Dataset](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder)** — 16,185 images across 196 classes (car make, model, and year), split ~50/50 between train and test. Organized by class folder, loaded via `torchvision.datasets.ImageFolder`.

```
stanford_cars/car_data/car_data/
├── train/
│   ├── Acura Integra Type R 2001/
│   ├── BMW M3 Coupe 2012/
│   └── ... (196 classes)
└── test/
    └── ... (same structure)
```

> **Known limitation:** the dataset covers primarily US/European market vehicles from roughly 1991–2012. It does not include current-generation or region-specific models (e.g. current Indian-market cars), so predictions on out-of-distribution vehicles will be low-confidence or incorrect — this is expected behavior, not a bug.

---

## 🚀 Quick Start

### Try it instantly (no setup)

👉 **[Open the live Streamlit demo](https://vehicle-detection-classification-mt3cj32atsaqx2qinw2mrn.streamlit.app)** and upload any car photo.

### Run it yourself

```bash
git clone https://github.com/Sahilmore469/vehicle-detection-classification.git
cd vehicle-detection-classification
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Retrain from scratch

Open `vehicle_detection_train_and_export.ipynb` in Google Colab (GPU runtime), run top to bottom. Produces `resnet50_cars_progressive.pt` and `class_names.json`, ready to redeploy.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Detection | YOLOv8 (Ultralytics) |
| Classification | PyTorch, ResNet50 (torchvision) |
| Preprocessing | OpenCV, PIL |
| Data handling | torchvision `ImageFolder`, JSON |
| Evaluation | scikit-learn (precision/recall/F1) |
| Interpretability | Grad-CAM |
| Visualization | Matplotlib |
| Web app | Streamlit |
| Hosting | Streamlit Community Cloud |

---

## 📁 Project Structure

```
.
├── vehicle_detection_train_and_export.ipynb   # full training pipeline, Colab-ready
├── streamlit_app.py                           # deployed web app
├── requirements.txt                           # Python dependencies
├── packages.txt                                # system dependencies (OpenCV support)
├── resnet50_cars_progressive.pt                # trained classifier weights
├── class_names.json                            # 196 class labels, in training order
└── README.md
```

---

## 🔮 Future Work

- [ ] Fine-tune YOLOv8 directly on vehicle imagery for tighter boxes
- [ ] Add multi-object tracking (ByteTrack) for smoother video inference
- [ ] Expand training data to cover current-generation/region-specific vehicles
- [ ] Quantize the classifier (ONNX/INT8) for faster CPU inference

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by Sahil More** · [GitHub](https://github.com/Sahilmore469) · [Live Demo](https://vehicle-detection-classification-mt3cj32atsaqx2qinw2mrn.streamlit.app)

</div>
