"""
MangaScourX - Advanced Multi-Scale Geometric-Aware Inpainting & Hybrid Text Detection
"""

from ._version import __version__

# Core utilities
from .core import (
    euclidean_distance_transform,
    connected_components,
    structure_tensor,
    perona_malik_diffusion,
    curvature_diffusion,
    PriorityQueue,
)

# Detection subsystem
from .detection import (
    BaseDetector,
    Detector,
    validate_image,
    merge_labeled,
    merge_binary,
    cleanup_mask,
    process_masks,
    detect_bubbles,
    clean_noise,
    detect_text_regions,
    stroke_width_transform,
    swt_to_mask,
    group_swt_components,
    CRAFTDetector,
)

# Inpainting subsystem
from .inpainting import (
    Inpainter,
    TeleaInpainter,
    PatchMatchInpainter,
    CoherenceTransport,
)

# Pipelines (high-level API)
from .pipelines import (
    MangaCleanPipeline,
    TextRemovePipeline,
)

__all__ = [
    # Version
    "__version__",
    
    # Core
    "euclidean_distance_transform",
    "connected_components",
    "structure_tensor",
    "perona_malik_diffusion",
    "curvature_diffusion",
    "PriorityQueue",
    
    # Detection
    "BaseDetector",
    "Detector",
    "validate_image",
    "merge_labeled",
    "merge_binary",
    "cleanup_mask",
    "process_masks",
    "detect_bubbles",
    "clean_noise",
    "detect_text_regions",
    "stroke_width_transform",
    "swt_to_mask",
    "group_swt_components",
    "CRAFTDetector",
    
    # Inpainting
    "Inpainter",
    "TeleaInpainter",
    "PatchMatchInpainter",
    "CoherenceTransport",
    
    # Pipelines
    "MangaCleanPipeline",
    "TextRemovePipeline",
]
