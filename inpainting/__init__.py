from .base import Inpainter
from .telea import TeleaInpainter
from .coherence import CoherenceTransport
from .patchmatch import PatchMatchInpainter

__all__ = [
    "Inpainter",
    "TeleaInpainter",
    "CoherenceTransport",
    "PatchMatchInpainter",
]
