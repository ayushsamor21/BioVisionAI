#!/usr/bin/env python3
"""Single-image ensemble inference with optional model-wise Grad-CAM."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from utils.config import load_config
from data.augmentation import get_val_augmentation
from models.classification.backbone_classifier import BackboneClassifier
from explainability.gradcam import run_grad_cam, save_heatmap_and_overlay


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RISK_HIGH = ["mel", "bcc"]  # consider melanoma and BCC as higher risk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="predict_outputs")
    parser.add_argument("--no_gradcam", action="store_true", help="Skip Grad-CAM")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf_cfg = config["classification"]

    manifest = torch.load(args.checkpoint, map_location="cpu")
    fold_checkpoints = manifest.get("fold_checkpoints", [])
    if not fold_checkpoints:
        raise ValueError("Checkpoint must be an ensemble manifest with fold checkpoints.")
    model_weights = manifest.get("model_weights", {})
    model_defs = manifest.get("model_defs", [])
    checkpoint_lookup = {}
    for item in fold_checkpoints:
        checkpoint_lookup.setdefault(item["backbone"], []).append(item["checkpoint"])

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_val_augmentation(
        image_size=(config["data"]["image_size"], config["data"]["image_size"]),
        use_hair_removal=config.get("preprocessing", {}).get("use_hair_removal", False),
    )
    transformed = transform(image=image_rgb)
    x = transformed["image"].unsqueeze(0).to(device)

    model_probs = {}
    with torch.no_grad():
        for backbone in model_defs:
            ckpts = checkpoint_lookup.get(backbone, [])
            if not ckpts:
                continue
            fold_probs = []
            for ckpt_path in ckpts:
                model = BackboneClassifier(
                    num_classes=clf_cfg["num_classes"],
                    backbone=backbone,
                    pretrained=False,
                ).to(device)
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
                model.eval()
                logits = model(x)
                fold_probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
            model_probs[backbone] = np.mean(fold_probs, axis=0)

    if not model_probs:
        raise RuntimeError("No models could be loaded from ensemble manifest.")
    weighted_prob = np.zeros((clf_cfg["num_classes"],), dtype=np.float32)
    total_weight = 0.0
    for backbone, probs in model_probs.items():
        weight = float(model_weights.get(backbone, 1.0))
        weighted_prob += weight * probs
        total_weight += weight
    probs = weighted_prob / max(total_weight, 1e-8)
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    risk = "high" if pred_class in RISK_HIGH else ("intermediate" if pred_class in ["akiec", "bkl"] else "low")

    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Risk category: {risk}")
    print("Probabilities:", dict(zip(CLASS_NAMES, [f"{p:.4f}" for p in probs])))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_gradcam:
        img_224 = cv2.resize(image_rgb, (config["data"]["image_size"], config["data"]["image_size"]))
        for backbone in model_defs:
            ckpts = checkpoint_lookup.get(backbone, [])
            if not ckpts:
                continue
            model = BackboneClassifier(
                num_classes=clf_cfg["num_classes"],
                backbone=backbone,
                pretrained=False,
            ).to(device)
            ckpt = torch.load(ckpts[0], map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.eval()
            heatmap = run_grad_cam(model, x, target_class=pred_idx, use_cuda=(device.type == "cuda"))
            h_path, o_path = save_heatmap_and_overlay(
                img_224, heatmap, out_dir, prefix=f"{Path(args.image).stem}_{backbone}"
            )
            print(f"[{backbone}] Heatmap: {h_path}, Overlay: {o_path}")


if __name__ == "__main__":
    main()
