"""
Base class for all inpainting algorithms.

Every inpainter **must** subclass `Inpainter` and implement `run()`.
Shared utilities (_find_boundary, _propagate) live here so that
algorithm subclasses stay thin.
"""

from __future__ import annotations

import abc
import numpy as np
from numpy.typing import NDArray


class Inpainter(abc.ABC):
    """
    Abstract base class for image inpainting algorithms.

    Contract
    --------
    - `run(image, mask)` receives a BGR uint8 image and a boolean/uint8
      mask (non-zero = hole to fill) and **must** return a uint8 image.
    - `_find_boundary` and `_propagate` are shared utilities available
      to every subclass (used by Telea, Coherence, etc.).
    """

    @abc.abstractmethod
    def run(
        self,
        image: NDArray[np.uint8],
        mask: NDArray,
    ) -> NDArray[np.uint8]:
        """
        Fill masked holes in `image`.

        Args:
            image : Input image (BGR uint8, shape H×W×C).
            mask  : Binary mask – non-zero pixels are holes to inpaint.

        Returns:
            Inpainted image (uint8, same shape as input).
        """
        ...

    # ------------------------------------------------------------------
    # Bug 7 Fix: _find_boundary and _propagate were called by telea.py
    # via self.* but were never defined anywhere.
    # ------------------------------------------------------------------

    @staticmethod
    def _find_boundary(mask: NDArray) -> list[tuple[int, int]]:
        """
        Return all pixels that sit *inside* the mask but are adjacent
        to at least one known (non-masked) pixel.

        These pixels form the propagation frontier for distance-based
        inpainting algorithms (Telea, FMM).

        Args:
            mask: 2-D array; non-zero values are the hole region.

        Returns:
            List of (y, x) tuples.
        """
        h, w = mask.shape
        boundary: list[tuple[int, int]] = []
        for y in range(h):
            for x in range(w):
                if not mask[y, x]:
                    continue
                # Pixel is in the hole – check 4-connected neighbours
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx]:
                        boundary.append((y, x))
                        break
        return boundary

    def _propagate(
        self,
        result: NDArray[np.float32],
        mask: NDArray,
        dist: NDArray[np.float32],
        pq,                         # PriorityQueue instance
        y: int,
        x: int,
    ) -> None:
        """
        Fill `result[y, x]` from its known neighbours using a
        distance-weighted average, then push any newly reachable
        masked neighbours onto the priority queue.

        This implements the core step of the Fast-Marching / Telea
        inpainting propagation.

        Args:
            result : Float image being built up (modified in-place).
            mask   : Hole mask (non-zero = hole).
            dist   : Distance map (modified in-place for neighbours).
            pq     : PriorityQueue – push (dist, y, x) tuples.
            y, x   : Coordinates of the pixel being filled.
        """
        h, w = mask.shape
        ndim = result.ndim
        c = result.shape[2] if ndim == 3 else 1

        # ── weighted average from known 8-connected neighbours ──────
        accum = np.zeros(c, dtype=np.float64)
        weight_total = 0.0

        neighbours_8 = (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        )
        for dy, dx in neighbours_8:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if mask[ny, nx]:
                continue                          # skip other holes

            eucl = np.sqrt(float(dy * dy + dx * dx))
            # Telea weight: closer known pixel + smaller existing dist
            w_val = 1.0 / (eucl * (1.0 + dist[ny, nx]) + 1e-8)

            if ndim == 3:
                accum += w_val * result[ny, nx].astype(np.float64)
            else:
                accum[0] += w_val * float(result[ny, nx])
            weight_total += w_val

        if weight_total > 0.0:
            filled = accum / weight_total
            if ndim == 3:
                result[y, x] = filled.astype(np.float32)
            else:
                result[y, x] = float(filled[0])

        # ── propagate distance to 4-connected masked neighbours ─────
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not mask[ny, nx]:
                continue                          # already known

            new_dist = dist[y, x] + 1.0          # unit step
            if new_dist < dist[ny, nx]:
                dist[ny, nx] = new_dist
                pq.push(new_dist, ny, nx)
