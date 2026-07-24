"""
FastAPI inference API for BIOVISION-AI.
POST /predict: image + optional metadata -> predicted class, confidence, risk, heatmap path.
"""

import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.augmentation import get_val_augmentation
from models.classification.backbone_classifier import BackboneClassifier
from explainability.gradcam import run_grad_cam, save_heatmap_and_overlay


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RISK_MAP = {"mel": "high", "bcc": "high", "akiec": "intermediate", "bkl": "intermediate", "df": "low", "nv": "low", "vasc": "low"}

app = FastAPI(title="BIOVISION-AI", description="Skin lesion classification API", version="1.0.0")

# Global model and device
_models = {}
_model_weights = {}
_device = None
_heatmap_dir = Path("api_heatmaps")


def load_model(checkpoint_path: str):
    global _models, _model_weights, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = torch.load(checkpoint_path, map_location="cpu")
    _models = {}
    _model_weights = manifest.get("model_weights", {})
    for item in manifest.get("fold_checkpoints", []):
        backbone = item["backbone"]
        model = BackboneClassifier(num_classes=7, backbone=backbone, pretrained=False).to(_device)
        ckpt = torch.load(item["checkpoint"], map_location=_device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        _models.setdefault(backbone, []).append(model)
    _heatmap_dir.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup():
    import os
    ckpt = os.environ.get("BIOVISION_CHECKPOINT", "checkpoints/classification/ensemble_final.pt")
    if Path(ckpt).exists():
        load_model(ckpt)
    # If no checkpoint, /predict will return 503


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": bool(_models)}


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    risk_category: str
    probabilities: dict
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    age: Optional[int] = Form(None),
    sex: Optional[str] = Form(None),
    body_location: Optional[str] = Form(None),
):
    """
    Run classification on uploaded skin lesion image.
    Optionally pass metadata (age, sex, body_location) for logging; not used by model yet.
    """
    if not _models:
        raise HTTPException(status_code=503, detail="Model not loaded. Set BIOVISION_CHECKPOINT or place ensemble_final.pt.")

    contents = await image.read()
    npy = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npy, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_val_augmentation(image_size=(224, 224))
    transformed = transform(image=img_rgb)
    x = transformed["image"].unsqueeze(0).to(_device)

    with torch.no_grad():
        weighted = np.zeros((7,), dtype=np.float32)
        total_weight = 0.0
        for backbone, models in _models.items():
            fold_prob = []
            for model in models:
                probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
                fold_prob.append(probs)
            avg_prob = np.mean(fold_prob, axis=0)
            w = float(_model_weights.get(backbone, 1.0))
            weighted += w * avg_prob
            total_weight += w
        probs = weighted / max(total_weight, 1e-8)
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    risk = RISK_MAP.get(pred_class, "low")
    probabilities = {c: float(p) for c, p in zip(CLASS_NAMES, probs)}

    heatmap_path = None
    overlay_path = None
    try:
        # Visualize Grad-CAM from first ensemble model for lightweight API response.
        first_backbone = next(iter(_models))
        cam_model = _models[first_backbone][0]
        heatmap = run_grad_cam(cam_model, x, target_class=pred_idx, use_cuda=(_device.type == "cuda"))
        img_224 = cv2.resize(img_rgb, (224, 224))
        prefix = Path(image.filename or "upload").stem
        h_path, o_path = save_heatmap_and_overlay(img_224, heatmap, _heatmap_dir, prefix=f"{prefix}_{first_backbone}")
        heatmap_path = h_path
        overlay_path = o_path
    except Exception:
        pass

    return PredictResponse(
        predicted_class=pred_class,
        confidence=confidence,
        risk_category=risk,
        probabilities=probabilities,
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
    )


def create_app(checkpoint_path: Optional[str] = None):
    if checkpoint_path and Path(checkpoint_path).exists():
        load_model(checkpoint_path)
    return app
