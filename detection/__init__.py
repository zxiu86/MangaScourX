"""
mangascourx - Advanced Multi-Scale Geometric-Aware Inpainting & Hybrid Text Detection
"""

from ._version import __version__

from .core import (
    euclidean_distance_transform,
    connected_components,
    structure_tensor,
    perona_malik_diffusion,
    curvature_diffusion,
    PriorityQueue,
)

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

from .inpainting import (
    Inpainter,
    TeleaInpainter,
    PatchMatchInpainter,
    CoherenceTransport,
)

from .pipelines import (
    MangaCleanPipeline,
    TextRemovePipeline,
)

__all__ = [
    "__version__",
    "euclidean_distance_transform",
    "connected_components",
    "structure_tensor",
    "perona_malik_diffusion",
    "curvature_diffusion",
    "PriorityQueue",
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
    "Inpainter",
    "TeleaInpainter",
    "PatchMatchInpainter",
    "CoherenceTransport",
    "MangaCleanPipeline",
    "TextRemovePipeline",
]
