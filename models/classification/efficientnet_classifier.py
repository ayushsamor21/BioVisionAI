"""Backward-compatible wrapper around generic backbone classifier."""

from models.classification.backbone_classifier import BackboneClassifier


class SkinLesionClassifier(BackboneClassifier):
    """Compatibility alias for previous imports."""
