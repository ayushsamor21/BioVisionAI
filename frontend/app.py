"""
Streamlit demo UI for BIOVISION-AI.
Upload image -> show prediction, probabilities, Grad-CAM heatmap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st
import torch

from data.augmentation import get_val_augmentation
from models.classification.backbone_classifier import BackboneClassifier
from explainability.gradcam import run_grad_cam, get_gradcam_heatmap_overlay


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratosis",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus (melanocytic)",
    "vasc": "Vascular lesion",
}
RISK_MAP = {"mel": "high", "bcc": "high", "akiec": "intermediate", "bkl": "intermediate", "df": "low", "nv": "low", "vasc": "low"}


@st.cache_resource
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = torch.load(checkpoint_path, map_location="cpu")
    models = {}
    for item in manifest.get("fold_checkpoints", []):
        model = BackboneClassifier(num_classes=7, backbone=item["backbone"], pretrained=False).to(device)
        ckpt = torch.load(item["checkpoint"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        models.setdefault(item["backbone"], []).append(model)
    return models, manifest.get("model_weights", {}), device


def main():
    st.set_page_config(page_title="BIOVISION-AI", page_icon="🔬", layout="wide")
    st.title("BIOVISION-AI: Skin Lesion Classification")
    st.markdown("Upload a dermoscopic image for **decision-support** classification. This is not a diagnostic device.")

    checkpoint = "checkpoints/classification/ensemble_final.pt"
    if not Path(checkpoint).exists():
        st.warning(f"Model checkpoint not found at `{checkpoint}`. Train first with `python scripts/train.py --config configs/default.yaml`.")
        st.stop()
    else:
        models, model_weights, device = load_model(checkpoint)

    uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Could not decode image.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_val_augmentation(image_size=(224, 224))
    transformed = transform(image=image_rgb)
    x = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        weighted = np.zeros((7,), dtype=np.float32)
        total_weight = 0.0
        for backbone, fold_models in models.items():
            fold_probs = []
            for model in fold_models:
                fold_probs.append(torch.softmax(model(x), dim=1).cpu().numpy()[0])
            p = np.mean(fold_probs, axis=0)
            w = float(model_weights.get(backbone, 1.0))
            weighted += w * p
            total_weight += w
        probs = weighted / max(total_weight, 1e-8)
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    risk = RISK_MAP.get(pred_class, "low")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Uploaded image", use_container_width=True)
    with col2:
        st.subheader("Prediction")
        st.metric("Predicted class", f"{pred_class} ({CLASS_DESCRIPTIONS.get(pred_class, pred_class)})")
        st.metric("Confidence", f"{confidence:.2%}")
        st.metric("Risk category", risk.upper())

    st.subheader("Class probabilities")
    for i, (c, p) in enumerate(zip(CLASS_NAMES, probs)):
        st.progress(float(p), text=f"{CLASS_DESCRIPTIONS.get(c, c)}: {p:.2%}")

    st.subheader("Grad-CAM explanation")
    try:
        first_backbone = next(iter(models))
        model = models[first_backbone][0]
        heatmap = run_grad_cam(model, x, target_class=pred_idx, use_cuda=(device.type == "cuda"))
        img_224 = cv2.resize(image_rgb, (224, 224))
        overlay = get_gradcam_heatmap_overlay(img_224, heatmap)
        st.image(overlay, caption="Grad-CAM overlay (areas that influenced the prediction)", use_container_width=True)
    except Exception as e:
        st.caption(f"Grad-CAM not available: {e}. Install: pip install grad-cam")

    st.caption("BIOVISION-AI is for educational and decision-support use only. Always follow clinical guidelines.")


if __name__ == "__main__":
    main()
