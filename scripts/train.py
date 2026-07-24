#!/usr/bin/env python3
"""K-Fold ensemble training for merged HAM10000 + ISIC datasets."""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from utils.config import load_config
from utils.logger import setup_logger
from data.datasets.skin_lesion import build_merged_dataframe, create_fold_dataloaders, create_kfold_indices
from models.classification.backbone_classifier import BackboneClassifier
from training.losses import FocalLoss
from evaluation.metrics import compute_classification_metrics


CLASS_NAMES = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]


def _build_loss(clf_cfg: dict, class_weights: torch.Tensor | None):
    if clf_cfg.get("loss", "focal") == "focal":
        return FocalLoss(gamma=clf_cfg.get("focal_gamma", 2.0), weight=class_weights)
    return nn.CrossEntropyLoss(weight=class_weights)


def _evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return compute_classification_metrics(y_true, y_pred, y_prob=y_prob), y_prob, y_true


def train_kfold_ensemble(config: dict):
    logger = setup_logger("train")
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    clf_cfg = config["classification"]
    ensemble_cfg = config.get("ensemble", {})
    backbones = ensemble_cfg.get("models", ["efficientnet_b0", "convnext_tiny"])
    n_splits = int(clf_cfg.get("k_folds", 5))
    use_amp = bool(clf_cfg.get("mixed_precision", True) and device.type == "cuda")
    image_size = (data_cfg.get("image_size", 224), data_cfg.get("image_size", 224))
    merged_df = build_merged_dataframe(data_cfg["root"], data_cfg.get("sources"))
    fold_splits = create_kfold_indices(merged_df["label"].tolist(), n_splits=n_splits, seed=config.get("seed", 42))
    checkpoint_dir = Path(clf_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    fold_scores = {m: [] for m in backbones}
    fold_artifacts = []
    for fold_id, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        logger.info(f"Starting fold {fold_id}/{n_splits}")
        train_loader, val_loader, _, _ = create_fold_dataloaders(
            merged_df=merged_df,
            train_indices=train_idx,
            val_indices=val_idx,
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg.get("num_workers", 4),
            image_size=image_size,
            use_hair_removal=config.get("preprocessing", {}).get("use_hair_removal", False),
            oversample=bool(clf_cfg.get("oversample", False)),
        )
        train_label_list = merged_df.iloc[train_idx]["label"].tolist()
        counts = Counter(train_label_list)
        total = sum(counts.values())
        class_weights = None
        if clf_cfg.get("use_class_weights", True):
            class_weights = torch.tensor(
                [total / (clf_cfg["num_classes"] * counts.get(i, 1)) for i in range(clf_cfg["num_classes"])],
                dtype=torch.float32,
                device=device,
            )

        for backbone in backbones:
            logger.info(f"Training {backbone} on fold {fold_id}")
            model = BackboneClassifier(
                num_classes=clf_cfg["num_classes"],
                backbone=backbone,
                pretrained=clf_cfg.get("pretrained", True),
                dropout=clf_cfg.get("dropout", 0.3),
            ).to(device)
            optimizer = AdamW(model.parameters(), lr=float(clf_cfg["lr"]), weight_decay=float(clf_cfg.get("weight_decay", 0.01)))
            scheduler_name = clf_cfg.get("scheduler", "cosine")
            if scheduler_name == "plateau":
                scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=clf_cfg["epochs"])
            criterion = _build_loss(clf_cfg, class_weights)
            scaler = GradScaler(enabled=use_amp)
            best_score = -1.0
            patience = clf_cfg.get("early_stopping_patience", 8)
            bad_epochs = 0
            best_path = checkpoint_dir / f"{backbone}_fold{fold_id}_best.pt"
            for epoch in range(clf_cfg["epochs"]):
                model.train()
                running_loss = 0.0
                for batch in train_loader:
                    x = batch["image"].to(device, non_blocking=True)
                    y = batch["label"].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(enabled=use_amp):
                        logits = model(x)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += float(loss.item())
                val_metrics, _, _ = _evaluate(model, val_loader, device)
                val_score = val_metrics.get("roc_auc_weighted", val_metrics["f1_weighted"])
                if scheduler_name == "plateau":
                    scheduler.step(val_score)
                else:
                    scheduler.step()
                logger.info(
                    f"[{backbone}] fold={fold_id} epoch={epoch+1}/{clf_cfg['epochs']} "
                    f"loss={running_loss / max(len(train_loader), 1):.4f} "
                    f"acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1_weighted']:.4f} auc={val_metrics.get('roc_auc_weighted', 0.0):.4f}"
                )
                if val_score > best_score:
                    best_score = val_score
                    bad_epochs = 0
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "backbone": backbone,
                            "fold": fold_id,
                            "val_metrics": val_metrics,
                            "config": config,
                        },
                        best_path,
                    )
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        logger.info(f"Early stopping for {backbone} fold {fold_id}")
                        break
            fold_scores[backbone].append(float(best_score))
            fold_artifacts.append({"backbone": backbone, "fold": fold_id, "checkpoint": str(best_path)})

    model_weights = {}
    if ensemble_cfg.get("weighted_voting", True):
        score_sums = {k: max(sum(v), 1e-8) for k, v in fold_scores.items()}
        total_sum = max(sum(score_sums.values()), 1e-8)
        model_weights = {k: float(v / total_sum) for k, v in score_sums.items()}
    else:
        uniform = 1.0 / max(len(backbones), 1)
        model_weights = {k: uniform for k in backbones}

    ensemble_path = checkpoint_dir / "ensemble_final.pt"
    torch.save(
        {
            "type": "kfold_ensemble",
            "model_defs": backbones,
            "num_classes": clf_cfg["num_classes"],
            "fold_checkpoints": fold_artifacts,
            "model_weights": model_weights,
            "config": config,
        },
        ensemble_path,
    )
    (checkpoint_dir / "kfold_scores.json").write_text(json.dumps(fold_scores, indent=2))
    logger.info(f"Saved ensemble manifest to {ensemble_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root

    train_kfold_ensemble(config)


if __name__ == "__main__":
    main()
