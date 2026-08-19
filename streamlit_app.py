"""
Vehicle Detection & Fine-Grained Classification — Streamlit app
Two-stage pipeline: YOLOv8 (detection) -> fine-tuned ResNet50 (196-class make/model/year classification)

Run locally with:  streamlit run streamlit_app.py
Deploy for free at: https://share.streamlit.io
"""

import io
import json
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet50
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config — EDIT THIS if your class list or weights filename differs
# ---------------------------------------------------------------------------
WEIGHTS_PATH = "resnet50_cars_progressive.pt"
CLASS_NAMES_PATH = "class_names.json"
VEHICLE_CLASSES = {2, 3, 5, 7}  # COCO ids: car, motorcycle, bus, truck

with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load models — cached so they only load once per session, not on every interaction
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detector():
    return YOLO("yolov8n.pt")


@st.cache_resource
def load_classifier():
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


detector = load_detector()
classifier = load_classifier()

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def detect_and_classify(image, conf=0.4):
    results = detector.predict(image, conf=conf, verbose=False)
    predictions = []

    for box in results[0].boxes:
        if int(box.cls) not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        crop = image.crop((x1, y1, x2, y2))
        tensor = val_transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = classifier(tensor)
            pred_idx = logits.argmax(1).item()
            conf_score = torch.softmax(logits, dim=1)[0, pred_idx].item()

        label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
        predictions.append({"box": (x1, y1, x2, y2), "class": label, "confidence": conf_score})

    return predictions


def annotate_image(image, predictions):
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)
    for pred in predictions:
        x1, y1, x2, y2 = pred["box"]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{pred['class']} ({pred['confidence']:.2f})",
                 color="white", fontsize=9, bbox=dict(facecolor="green", alpha=0.7))
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Vehicle Detection & Classification", page_icon="🚗", layout="wide")

st.title("🚗 Vehicle Detection & Fine-Grained Classification")
st.write(
    "Two-stage computer vision pipeline: **YOLOv8** detects vehicles, then a **fine-tuned ResNet50** "
    "classifies each one into 1 of 196 make/model/year classes (Stanford Cars dataset)."
)

conf_threshold = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.4, 0.05)
uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    with st.spinner("Running detection + classification..."):
        preds = detect_and_classify(image, conf=conf_threshold)

    with col2:
        st.subheader("Detections")
        if not preds:
            st.warning("No vehicles detected. Try lowering the confidence threshold in the sidebar.")
            st.image(image, use_container_width=True)
        else:
            annotated = annotate_image(image, preds)
            st.image(annotated, use_container_width=True)

    if preds:
        st.subheader("Predictions")
        for p in preds:
            st.write(f"**{p['class']}** — confidence: {p['confidence']:.2%}")
else:
    st.info("Upload an image to get started.")
