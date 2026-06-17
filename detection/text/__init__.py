"""
Text detection - MSER, SWT, and CRAFT-based text localization.
"""

from .mser import detect_text_regions
from .swt import (
    stroke_width_transform,
    swt_to_mask,
    group_swt_components,
)
from .craft_adapter import CRAFTDetector

__all__ = [
    "detect_text_regions",
    "stroke_width_transform",
    "swt_to_mask",
    "group_swt_components",
    "CRAFTDetector",
]
