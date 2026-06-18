"""
Text detection subpackage – MSER, SWT, CRAFT adapters.
"""

from .mser import detect_text_regions
from .craft_adapter import CRAFTDetector

__all__ = [
    "detect_text_regions",
    "CRAFTDetector",
]
