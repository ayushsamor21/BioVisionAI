#!/usr/bin/env python3
"""Evaluate k-fold ensemble with out-of-fold predictions."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from utils.config import load_config
from data.datasets.skin_lesion import build_merged_dataframe, create_fold_dataloaders, create_kfold_indices
from models.classification.backbone_classifier import BackboneClassifier
from evaluation.metrics import compute_classification_metrics, classification_report_dict
from evaluation.plots import plot_confusion_matrix, plot_roc_curves, plot_pr_curves


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ensemble manifest (ensemble_final.pt)")
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs")
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    clf_cfg = config["classification"]
    manifest = torch.load(args.checkpoint, map_location="cpu")
    fold_checkpoints = manifest["fold_checkpoints"]
    model_weights = manifest.get("model_weights", {})
    merged_df = build_merged_dataframe(data_cfg["root"], data_cfg.get("sources"))
    splits = create_kfold_indices(
        merged_df["label"].tolist(), n_splits=int(clf_cfg.get("k_folds", 5)), seed=config.get("seed", 42)
    )

    all_probs = np.zeros((len(merged_df), clf_cfg["num_classes"]), dtype=np.float32)
    all_labels = merged_df["label"].to_numpy()
    checkpoint_lookup = {(item["backbone"], item["fold"]): item["checkpoint"] for item in fold_checkpoints}

    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        _, val_loader, _, _ = create_fold_dataloaders(
            merged_df=merged_df,
            train_indices=train_idx,
            val_indices=val_idx,
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg.get("num_workers", 4),
            image_size=(data_cfg["image_size"], data_cfg["image_size"]),
            use_hair_removal=config.get("preprocessing", {}).get("use_hair_removal", False),
            oversample=False,
        )
        fold_prob = np.zeros((len(val_idx), clf_cfg["num_classes"]), dtype=np.float32)
        weight_sum = 0.0
        for backbone in manifest["model_defs"]:
            ckpt_path = checkpoint_lookup[(backbone, fold_id)]
            model = BackboneClassifier(num_classes=clf_cfg["num_classes"], backbone=backbone, pretrained=False).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.eval()
            probs_parts = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(device)
                    probs = torch.softmax(model(x), dim=1).cpu().numpy()
                    probs_parts.append(probs)
            model_prob = np.concatenate(probs_parts, axis=0)
            weight = float(model_weights.get(backbone, 1.0))
            fold_prob += weight * model_prob
            weight_sum += weight
        if weight_sum > 0:
            fold_prob /= weight_sum
        all_probs[val_idx] = fold_prob

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = y_prob.argmax(axis=1)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names=CLASS_NAMES)
    print("Metrics:", metrics)
    report = classification_report_dict(y_true, y_pred, CLASS_NAMES)
    print("Classification report:", report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASS_NAMES, save_path=str(out_dir / "confusion_matrix.png"))
    plot_roc_curves(y_true, y_prob, CLASS_NAMES, save_path=str(out_dir / "roc_curves.png"))
    plot_pr_curves(y_true, y_prob, CLASS_NAMES, save_path=str(out_dir / "pr_curves.png"))
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
