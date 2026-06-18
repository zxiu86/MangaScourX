"""
Telea inpainting – Fast-Marching-based image inpainting.

Reference: Alexandru Telea, "An image inpainting technique based on
the fast marching method", Journal of graphics tools, 2004.

Uses `Inpainter` as its base class, which already provides
`_find_boundary()` and `_propagate()` so this file stays clean.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Bug 6 fix: base.py now exists in inpainting/ with the Inpainter class
from .base import Inpainter
# Bug: ..core was right – PriorityQueue lives in mangascourx/core/
from ..core.priority_queue import PriorityQueue


class TeleaInpainter(Inpainter):
    """
    Fast-Marching image inpainting (Telea 2004).

    Pixels are filled in order of increasing distance from the
    boundary of the known region (via a min-heap priority queue).
    Each unknown pixel receives a distance-weighted average of its
    known neighbours.
    """

    def __init__(self, radius: int = 5) -> None:
        """
        Args:
            radius: Neighbourhood radius used when collecting known
                    pixels for the weighted average (reserved for
                    future radius-based weighting; current impl uses
                    8-connected neighbours from the base class).
        """
        self.radius = radius

    # ------------------------------------------------------------------
    # Inpainter contract
    # ------------------------------------------------------------------

    def run(
        self,
        image: NDArray[np.uint8],
        mask: NDArray,
    ) -> NDArray[np.uint8]:
        """
        Inpaint `image` wherever `mask` is non-zero.

        Args:
            image : BGR uint8 image.
            mask  : 2-D mask – non-zero pixels are holes.

        Returns:
            Inpainted uint8 image (same shape as input).
        """
        result = image.astype(np.float32).copy()

        h, w = mask.shape

        # Distance of each pixel from the boundary (inf = deep inside hole)
        dist = np.full((h, w), np.inf, dtype=np.float32)

        # Min-heap ordered by distance
        pq = PriorityQueue(h * w)

        # Seed: pixels on the inner boundary of the hole have distance 0
        # Bug 7 fix: _find_boundary is now in Inpainter base class
        boundary = self._find_boundary(mask)
        for y, x in boundary:
            dist[y, x] = 0.0
            pq.push(0.0, y, x)

        # March from boundary inward
        while not pq.empty():
            d, y, x = pq.pop()

            # Bug 7 fix: _propagate is now in Inpainter base class
            self._propagate(result, mask, dist, pq, y, x)

        return np.clip(result, 0, 255).astype(np.uint8)
