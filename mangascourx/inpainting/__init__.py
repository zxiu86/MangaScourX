"""
Inpainting package – PatchMatch, Telea, and Coherence Transport algorithms.
"""

from .patchmatch.engine import PatchMatchInpainter
from .telea import TeleaInpainter
from .coherence import CoherenceTransport

__all__ = [
    "PatchMatchInpainter",
    "TeleaInpainter",
    "CoherenceTransport",
]
