"""Dataset utilities for merged HAM10000 + ISIC training."""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Default HAM10000 dx (diagnosis) to class index
# df(dermatofibroma) & vasc(vascular lesion) are rare, so they may be dropped in some experiments.(and may not be present in the chosen dataset)
DX_TO_IDX = { #Convert disease names into numbers.
    "akiec": 0,  # actinic keratosis
    "bcc": 1,    # basal cell carcinoma
    "bkl": 2,    # benign keratosis
    "df": 3,     # dermatofibroma
    "mel": 4,    # melanoma
    "nv": 5,     # nevus
    "vasc": 6,   # vascular
}

LABEL_ALIASES = { #Normalize different spellings.
    "melanoma": "mel",
    "mel": "mel",
    "melanocytic nevus": "nv",
    "nevus": "nv",
    "nv": "nv",
    "basal cell carcinoma": "bcc",
    "bcc": "bcc",
    "actinic keratosis": "akiec",
    "ak": "akiec",
    "akiec": "akiec",
    "benign keratosis": "bkl",
    "bkl": "bkl",
    "dermatofibroma": "df",
    "df": "df",
    "vascular lesion": "vasc",
    "vascular lesions": "vasc",
    "vasc": "vasc",
}


class SkinLesionDataset(Dataset):
    """
    Dataset for skin lesion classification (and optional segmentation).
    Supports CSV with image_id, dx (diagnosis). Image paths can be in a folder or in CSV.
    
    This class is responsible for creating ONE training sample.
    it asks pytorch to load one image and its label (and optional mask) from the dataset.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        mask_paths: Optional[List[Optional[str]]] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mask_paths = mask_paths  # same length as image_paths, None if no mask
        self.class_names = class_names or list(DX_TO_IDX.keys())

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        out = {"image": image, "label": label, "path": path}
        if self.mask_paths and self.mask_paths[idx]:
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.float32)
                out["mask"] = mask
            else:
                out["mask"] = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            out["mask"] = None

        if self.transform:
            if out["mask"] is not None and out["mask"].size > 0:
                transformed = self.transform(image=image, mask=out["mask"])
                out["image"] = transformed["image"]
                out["mask"] = transformed["mask"]
            else:
                transformed = self.transform(image=image)
                out["image"] = transformed["image"]
                if out["mask"] is not None:
                    out["mask"] = None  # drop mask if not transformed

        if not isinstance(out["image"], torch.Tensor):
            out["image"] = torch.from_numpy(np.asarray(out["image"]).transpose(2, 0, 1)).float()
        out["label"] = torch.tensor(out["label"], dtype=torch.long)
        if out.get("mask") is not None:
            m = out["mask"]
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m).float()
            if m.dim() == 2:
                m = m.unsqueeze(0)
            out["mask"] = m
        return out

#Metadata contains image name But Python needs full path to load the image. This function resolves the full path to the image based on the metadata row.(searches different folder until it finds the image)
def _resolve_image_path(root: Path, row: pd.Series, image_col: str, images_dir: str) -> str:
    """Get full path to image from metadata row."""
    if image_col in row and pd.notna(row[image_col]):
        p = root / row[image_col]
        if p.exists():
            return str(p)
        p2 = root / "images" / row[image_col]
        if p2.exists():
            return str(p2)
        p3 = root / images_dir / row[image_col]
        if p3.exists():
            return str(p3)
    # Try common patterns
    for folder in ["images", "ham10000_images", "ISIC", ""]:
        for ext in [".jpg", ".jpeg", ".png"]:
            cand = root / folder / (str(row.get("image_id", row.name)) + ext)
            if cand.exists():
                return str(cand)
    return ""

#Every dataset may use different label names, So every dataset becomes compatible.
def standardize_label(label: str) -> Optional[str]:
    key = str(label).strip().lower()
    return LABEL_ALIASES.get(key, key if key in DX_TO_IDX else None)


def _default_sources(data_root: Path) -> List[Dict[str, str]]:
    return [
        {
            "name": "ham10000",
            "root": str(data_root),
            "metadata_file": "HAM10000_metadata.csv",
            "image_column": "image_id",
            "label_column": "dx",
            "images_dir": "ham10000_images",
        },
        {
            "name": "isic",
            "root": str(data_root),
            "metadata_file": "ISIC_metadata.csv",
            "image_column": "image_id",
            "label_column": "dx",
            "images_dir": "images",
        },
    ]


def build_merged_dataframe(data_root: str | Path, sources: Optional[List[Dict[str, str]]] = None) -> pd.DataFrame:
    root = Path(data_root)
    srcs = sources or _default_sources(root)
    records = []
    for source in srcs:
        s_root = Path(source.get("root", root))
        metadata_file = source.get("metadata_file", "metadata.csv")
        metadata_path = s_root / metadata_file
        if not metadata_path.exists():
            continue
        df = pd.read_csv(metadata_path)
        image_col = source.get("image_column", "image_id")
        label_col = source.get("label_column", "dx")
        if image_col not in df.columns and "image_id" in df.columns:
            image_col = "image_id"
        if label_col not in df.columns:
            continue
        images_dir = source.get("images_dir", "images")
        for _, row in df.iterrows():
            raw_label = row.get(label_col)
            std_label = standardize_label(raw_label)
            if std_label is None:
                continue
            path = _resolve_image_path(s_root, row, image_col, images_dir)
            if not path or not os.path.exists(path):
                continue
            records.append(
                {
                    "image_path": path,
                    "label_name": std_label,
                    "label": DX_TO_IDX[std_label],
                    "source": source.get("name", metadata_file),
                }
            )
    if not records:
        raise ValueError("No valid images found across HAM10000/ISIC sources. Check metadata files and image folders.")
    merged = pd.DataFrame.from_records(records).drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return merged

#This helps fight class imbalance, diff disease have diff no. of images in the dataset so This sampler gives more importance to rare diseases.
def _make_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = np.bincount(np.array(labels))
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = [weights[y] for y in labels]
    return WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)


def create_kfold_indices(labels: List[int], n_splits: int = 5, seed: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(labels))
    return list(skf.split(idx, np.array(labels)))


def create_fold_dataloaders(
    merged_df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    use_hair_removal: bool = False,
    oversample: bool = False,
):
    from data.augmentation import get_train_augmentation, get_val_augmentation

    train_df = merged_df.iloc[train_indices].reset_index(drop=True)
    val_df = merged_df.iloc[val_indices].reset_index(drop=True)
    train_paths = train_df["image_path"].tolist()
    train_labels = train_df["label"].tolist()
    val_paths = val_df["image_path"].tolist()
    val_labels = val_df["label"].tolist()

    train_tf = get_train_augmentation(image_size=image_size, use_hair_removal=use_hair_removal)
    val_tf = get_val_augmentation(image_size=image_size, use_hair_removal=use_hair_removal)
    train_ds = SkinLesionDataset(train_paths, train_labels, transform=train_tf)
    val_ds = SkinLesionDataset(val_paths, val_labels, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=not oversample,
        sampler=_make_sampler(train_labels) if oversample else None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds, val_ds


def load_metadata_and_paths(
    data_root: str | Path,
    metadata_file: Optional[str] = None,
    image_column: str = "image_id",
    label_column: str = "dx",
    images_dir: str = "images",
) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """Backward-compatible single-source loader."""
    root = Path(data_root)
    meta_file = metadata_file or "HAM10000_metadata.csv"
    df = pd.read_csv(root / meta_file)
    all_paths, all_labels = [], []
    for _, row in df.iterrows():
        path = _resolve_image_path(root, row, image_column, images_dir)
        label = standardize_label(row.get(label_column, ""))
        if path and os.path.exists(path) and label in DX_TO_IDX:
            all_paths.append(path)
            all_labels.append(DX_TO_IDX[label])
    return df, all_paths, all_labels


def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    image_size: Tuple[int, int] = (224, 224),
    metadata_file: Optional[str] = None,
    mask_dir: Optional[str] = None,
    seed: int = 42,
    image_column: str = "image_id",
    label_column: str = "dx",
):
    """Legacy split dataloaders, now with stratified splitting."""
    from data.augmentation import get_train_augmentation, get_val_augmentation

    _, paths, labels = load_metadata_and_paths(
        Path(data_root), metadata_file=metadata_file, image_column=image_column, label_column=label_column
    )
    if not paths:
        raise ValueError(
            f"No valid images found in {data_root}. "
            "Ensure metadata.csv has 'image_id' and 'dx' columns and images exist in data/images/."
        )

    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels, test_size=(1.0 - train_ratio), stratify=labels, random_state=seed
    )
    relative_val_size = val_ratio / max((val_ratio + test_ratio), 1e-8)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths, rest_labels, test_size=(1.0 - relative_val_size), stratify=rest_labels, random_state=seed
    )

    def _mask_paths(img_paths: List[str], root: Path, mask_d: Optional[str]) -> List[Optional[str]]:
        if not mask_d:
            return [None] * len(img_paths)
        out = []
        for p in img_paths:
            base = Path(p).stem
            for ext in [".png", ".jpg"]:
                m = root / mask_d / (base + ext)
                if m.exists():
                    out.append(str(m))
                    break
            else:
                out.append(None)
        return out

    root = Path(data_root)
    train_masks = _mask_paths(train_paths, root, mask_dir)
    val_masks = _mask_paths(val_paths, root, mask_dir)
    test_masks = _mask_paths(test_paths, root, mask_dir)

    train_tf = get_train_augmentation(image_size=image_size)
    val_tf = get_val_augmentation(image_size=image_size)

    train_ds = SkinLesionDataset(train_paths, train_labels, transform=train_tf, mask_paths=train_masks)
    val_ds = SkinLesionDataset(val_paths, val_labels, transform=val_tf, mask_paths=val_masks)
    test_ds = SkinLesionDataset(test_paths, test_labels, transform=val_tf, mask_paths=test_masks)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
