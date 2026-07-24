"""Albumentations preprocessing/augmentation for dermoscopy images."""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List
from data.preprocessing import optional_hair_removal


class HairRemovalTransform(A.ImageOnlyTransform):
    """Optional DullRazor-style artifact suppression."""

    def apply(self, img, **params):
        return optional_hair_removal(img)


def get_train_augmentation(
    image_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    horizontal_flip: float = 0.5,
    vertical_flip: float = 0.5,
    rotate_limit: int = 30,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    random_crop_scale: Tuple[float, float] = (0.8, 1.0),
    use_hair_removal: bool = False,
) -> A.Compose:
    """Training augmentations for skin lesion images."""
    transforms = []
    if use_hair_removal:
        transforms.append(HairRemovalTransform(p=0.4))
    transforms.extend(
        [
            A.HorizontalFlip(p=horizontal_flip),
            A.VerticalFlip(p=vertical_flip),
            A.Rotate(limit=rotate_limit, border_mode=cv2.BORDER_REFLECT, p=0.7),
            A.RandomResizedCrop(
                height=image_size[0],
                width=image_size[1],
                scale=random_crop_scale,
                ratio=(0.8, 1.2),
                p=1.0,
            ),
            A.ColorJitter(brightness=brightness_limit, contrast=contrast_limit, saturation=0.2, hue=0.05, p=0.6),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def get_val_augmentation(
    image_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    use_hair_removal: bool = False,
) -> A.Compose:
    """Validation/test: resize + normalize only."""
    transforms = []
    if use_hair_removal:
        transforms.append(HairRemovalTransform(p=1.0))
    transforms.extend(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


