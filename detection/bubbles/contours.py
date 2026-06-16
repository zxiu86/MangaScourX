"""
Contour detection and processing tools for bubble masks.

Provides functions to:
- Detect contours in binary masks
- Compute convex hulls
- Extract bounding rectangles and circles
- Draw contours and hulls for visualization
- Filter contours by area

All functions expect binary masks (uint8, 0 and 255) unless stated otherwise.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_binary_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert input to a strict 0/255 binary mask (uint8)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


# ---------------------------------------------------------------------------
# Contour detection
# ---------------------------------------------------------------------------

def find_contours(
    mask: NDArray[np.uint8],
    mode: int = cv2.RETR_EXTERNAL,
    method: int = cv2.CHAIN_APPROX_SIMPLE,
) -> tuple[list[NDArray[np.int32]], NDArray[np.int32] | None]:
    """
    Find contours in a binary mask.

    Args:
        mask: Binary mask (0/255, uint8).
        mode: OpenCV contour retrieval mode (default: RETR_EXTERNAL).
        method: Contour approximation method (default: CHAIN_APPROX_SIMPLE).

    Returns:
        Tuple of (contours, hierarchy) where contours is a list of (N,1,2)
        arrays and hierarchy is a numpy array or None.

    Note:
        The function modifies the mask internally (OpenCV's findContours
        can alter the source image). A copy is used automatically.
    """
    mask = _ensure_binary_mask(mask)
    contours, hierarchy = cv2.findContours(mask.copy(), mode, method)
    # OpenCV 4+ returns (contours, hierarchy)
    # OpenCV 3 returns (image, contours, hierarchy) but we handle it via
    # the standard cv2.findContours signature in 4.x. For safety:
    if len(contours) == 2 and hierarchy is None:
        # Could be a 3.x style return; fix it
        _, contours, hierarchy = contours  # type: ignore[arg-type]
    return contours, hierarchy


def filter_contours_by_area(
    contours: list[NDArray[np.int32]],
    min_area: float = 0.0,
    max_area: float = float("inf"),
) -> list[NDArray[np.int32]]:
    """
    Filter contours by contour area.

    Args:
        contours: List of contours from find_contours.
        min_area: Minimum contour area (inclusive).
        max_area: Maximum contour area (inclusive).

    Returns:
        Filtered list of contours.
    """
    if min_area == 0.0 and max_area == float("inf"):
        return contours
    return [
        c
        for c in contours
        if min_area <= cv2.contourArea(c) <= max_area
    ]


# ---------------------------------------------------------------------------
# Convex hulls
# ---------------------------------------------------------------------------

def get_convex_hulls(
    contours: list[NDArray[np.int32]],
) -> list[NDArray[np.int32]]:
    """
    Compute convex hull for each contour.

    Args:
        contours: List of contours.

    Returns:
        List of hulls (same format as contours).
    """
    hulls = []
    for c in contours:
        hull = cv2.convexHull(c)
        if hull is not None:
            hulls.append(hull)
    return hulls


# ---------------------------------------------------------------------------
# Bounding shapes
# ---------------------------------------------------------------------------

def get_bounding_rects(
    contours: list[NDArray[np.int32]],
) -> list[tuple[int, int, int, int]]:
    """
    Compute upright bounding rectangles for contours.

    Args:
        contours: List of contours.

    Returns:
        List of (x, y, width, height) tuples.
    """
    return [cv2.boundingRect(c) for c in contours]


def get_bounding_circles(
    contours: list[NDArray[np.int32]],
) -> list[tuple[tuple[float, float], float]]:
    """
    Compute minimum enclosing circles for contours.

    Args:
        contours: List of contours.

    Returns:
        List of ((center_x, center_y), radius) tuples.
        Values are float.
    """
    circles = []
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        circles.append(((x, y), radius))
    return circles


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def draw_contours(
    image: NDArray[np.uint8],
    contours: list[NDArray[np.int32]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    inplace: bool = False,
) -> NDArray[np.uint8]:
    """
    Draw contours on an image.

    Args:
        image: Input image (BGR or grayscale). Must be uint8.
        contours: Contours to draw.
        color: BGR color tuple.
        thickness: Line thickness. Negative thickness fills the contour.
        inplace: If True, draw directly on input image (modified in-place).

    Returns:
        Image with drawn contours (same shape as input).
    """
    out = image if inplace else image.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def draw_convex_hulls(
    image: NDArray[np.uint8],
    hulls: list[NDArray[np.int32]],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    inplace: bool = False,
) -> NDArray[np.uint8]:
    """
    Draw convex hulls on an image.

    Args:
        image: Input image (BGR or grayscale).
        hulls: Convex hulls (list of point arrays).
        color: BGR color tuple.
        thickness: Line thickness.
        inplace: Modify image in-place if True.

    Returns:
        Image with drawn hulls.
    """
    out = image if inplace else image.copy()
    cv2.drawContours(out, hulls, -1, color, thickness)
    return out


def draw_bounding_rects(
    image: NDArray[np.uint8],
    rects: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    inplace: bool = False,
) -> NDArray[np.uint8]:
    """
    Draw bounding rectangles on an image.

    Args:
        image: Input image.
        rects: List of (x, y, w, h).
        color: BGR color.
        thickness: Line thickness.
        inplace: If True, modify input image.

    Returns:
        Image with rectangles drawn.
    """
    out = image if inplace else image.copy()
    for x, y, w, h in rects:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def draw_bounding_circles(
    image: NDArray[np.uint8],
    circles: list[tuple[tuple[float, float], float]],
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
    inplace: bool = False,
) -> NDArray[np.uint8]:
    """
    Draw minimum enclosing circles on an image.

    Args:
        image: Input image.
        circles: List of ((center_x, center_y), radius).
        color: BGR color.
        thickness: Line thickness.
        inplace: If True, modify input image.

    Returns:
        Image with circles drawn.
    """
    out = image if inplace else image.copy()
    for (cx, cy), radius in circles:
        center = (int(round(cx)), int(round(cy)))
        cv2.circle(out, center, int(radius), color, thickness)
    return out


# ---------------------------------------------------------------------------
# High‑level bubble detection pipeline
# ---------------------------------------------------------------------------

def detect_bubbles(
    mask: NDArray[np.uint8],
    min_area: float = 0.0,
    max_area: float = float("inf"),
) -> dict:
    """
    Full bubble detection from a binary mask.

    Steps:
        1. Find contours (external only)
        2. Filter by area
        3. Compute convex hulls
        4. Compute bounding rectangles and circles

    Args:
        mask: Binary mask (0/255, uint8).
        min_area: Minimum bubble area.
        max_area: Maximum bubble area.

    Returns:
        Dictionary with keys:
            - 'contours': list of filtered contours
            - 'hulls': convex hulls for those contours
            - 'bounding_rects': upright bounding rectangles
            - 'bounding_circles': minimum enclosing circles
    """
    mask = _ensure_binary_mask(mask)

    # Detect external contours only (bubble outer boundaries)
    contours, _ = find_contours(mask, mode=cv2.RETR_EXTERNAL)

    # Filter by size
    contours = filter_contours_by_area(contours, min_area, max_area)

    hulls = get_convex_hulls(contours)
    rects = get_bounding_rects(contours)
    circles = get_bounding_circles(contours)

    return {
        "contours": contours,
        "hulls": hulls,
        "bounding_rects": rects,
        "bounding_circles": circles,
    }