"""
Detection subsystem - text and bubble detection.
"""

from .base import BaseDetector, validate_image
from .detection import Detector
from .mask import merge_labeled, merge_binary, cleanup_mask, process_masks

# Bubbles subpackage
from .bubbles import (
    detect_bubbles,
    clean_noise,
)

# Text subpackage
from .text import (
    detect_text_regions,
    stroke_width_transform,
    swt_to_mask,
    group_swt_components,
    CRAFTDetector,
)

__all__ = [
    # Base
    "BaseDetector",
    "validate_image",
    
    # Orchestrator
    "Detector",
    
    # Mask processing
    "merge_labeled",
    "merge_binary",
    "cleanup_mask",
    "process_masks",
    
    # Bubbles
    "detect_bubbles",
    "clean_noise",
    
    # Text
    "detect_text_regions",
    "stroke_width_transform",
    "swt_to_mask",
    "group_swt_components",
    "CRAFTDetector",
]
