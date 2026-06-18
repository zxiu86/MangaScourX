"""
Stroke Width Transform (SWT) for complex text detection.

Based on the algorithm by Epshtein, Ofek, and Wexler (CVPR 2010).
SWT computes the width of the stroke for each pixel in the image,
which is a stable feature for detecting text regardless of font,
scale, or orientation.

Typical usage:
    from text.swt import stroke_width_transform, swt_to_mask

    swt_map = stroke_width_transform(gray_image)
    text_mask = swt_to_mask(swt_map, min_stroke=3, max_stroke=30)
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helper: edge detection with gradient direction
# ---------------------------------------------------------------------------

def _compute_edges_and_gradients(
    gray: NDArray[np.uint8],
    edge_low: float = 50.0,
    edge_high: float = 150.0,
    gradient_kernel_size: int = 3,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute Canny edge map and Sobel gradient directions.

    Args:
        gray: Grayscale uint8 image.
        edge_low: Low threshold for Canny.
        edge_high: High threshold for Canny.
        gradient_kernel_size: Sobel kernel size (odd).

    Returns:
        edge_map: Binary edge map (uint8, 0/255).
        gx: Sobel x derivative (float32).
        gy: Sobel y derivative (float32).
    """
    # Edge map
    edges = cv2.Canny(gray, edge_low, edge_high)

    # Gradient components
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=gradient_kernel_size)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=gradient_kernel_size)

    return edges, gx, gy


# ---------------------------------------------------------------------------
# Core SWT algorithm
# ---------------------------------------------------------------------------

def stroke_width_transform(
    gray: NDArray[np.uint8],
    edge_low: float = 50.0,
    edge_high: float = 150.0,
    max_stroke_width: int = 70,
    gradient_kernel_size: int = 3,
    direction_override: tuple[float, float] | None = None,
) -> NDArray[np.float32]:
    """
    Compute the Stroke Width Transform of a grayscale image.

    For every edge pixel, a ray is shot in the direction of the gradient
    (and opposite direction) until another edge pixel is found (or
    max_stroke_width is reached). All pixels along the ray receive the
    minimum stroke width value found.

    Args:
        gray: Grayscale image (uint8, single channel).
        edge_low: Canny low threshold.
        edge_high: Canny high threshold.
        max_stroke_width: Maximum stroke width to consider (speed/memory).
        gradient_kernel_size: Sobel kernel size for gradient computation.
        direction_override: If not None, (dx, dy) unit vector override for
            gradient direction (e.g. (-1, 0) for horizontal text).
            Useful for enforcing a known text orientation.

    Returns:
        SWT map (float32), where each pixel holds the estimated stroke
        width, or 0 if no stroke was found.
    """
    # Input validation
    if gray.ndim != 2:
        raise ValueError("Input must be a single-channel grayscale image.")

    edges, gx, gy = _compute_edges_and_gradients(
        gray, edge_low, edge_high, gradient_kernel_size
    )

    # Magnitude and direction (unit vectors)
    magnitude = np.sqrt(gx**2 + gy**2)
    mask_mag = magnitude > 1e-5  # avoid division by zero
    dx = np.zeros_like(gx)
    dy = np.zeros_like(gy)
    if direction_override is not None:
        dx[:], dy[:] = direction_override
    else:
        dx[mask_mag] = gx[mask_mag] / magnitude[mask_mag]
        dy[mask_mag] = gy[mask_mag] / magnitude[mask_mag]

    # Find edge pixel coordinates
    edge_pts = np.column_stack(np.where(edges > 0))  # (row, col)

    # SWT accumulator: store minimum stroke width per pixel
    height, width = gray.shape
    swt = np.full((height, width), np.inf, dtype=np.float32)

    max_steps = max_stroke_width

    for pt in edge_pts:
        row, col = pt
        if not mask_mag[row, col]:
            continue

        dr = dy[row, col]  # direction along rows (y-axis)
        dc = dx[row, col]  # direction along cols (x-axis)

        # Normalize to a step of 1 pixel length
        norm = np.sqrt(dr**2 + dc**2)
        if norm == 0:
            continue
        dr /= norm
        dc /= norm

        # Shoot ray in positive gradient direction
        ray_pixels = []
        found_pos = False
        for step in range(1, max_steps + 1):
            r = int(round(row + dr * step))
            c = int(round(col + dc * step))
            if r < 0 or r >= height or c < 0 or c >= width:
                break
            ray_pixels.append((r, c))
            if edges[r, c] > 0:
                # Check if gradient directions are roughly opposite
                if mask_mag[r, c]:
                    dot = dx[row, col] * dx[r, c] + dy[row, col] * dy[r, c]
                    # Opposite direction means dot ≈ -1
                    if dot < -0.5:  # ~cos(120°) – tolerant
                        found_pos = True
                        stroke_len = step
                        break
        if found_pos and ray_pixels:
            for r, c in ray_pixels[:stroke_len]:
                if stroke_len < swt[r, c]:
                    swt[r, c] = stroke_len

        # Shoot ray in negative gradient direction
        ray_pixels = []
        found_neg = False
        for step in range(1, max_steps + 1):
            r = int(round(row - dr * step))
            c = int(round(col - dc * step))
            if r < 0 or r >= height or c < 0 or c >= width:
                break
            ray_pixels.append((r, c))
            if edges[r, c] > 0:
                if mask_mag[r, c]:
                    dot = dx[row, col] * dx[r, c] + dy[row, col] * dy[r, c]
                    if dot < -0.5:
                        found_neg = True
                        stroke_len = step
                        break
        if found_neg and ray_pixels:
            for r, c in ray_pixels[:stroke_len]:
                if stroke_len < swt[r, c]:
                    swt[r, c] = stroke_len

    # Replace inf with 0
    swt[np.isinf(swt)] = 0

    return swt


# ---------------------------------------------------------------------------
# Post-processing: mask from SWT
# ---------------------------------------------------------------------------

def swt_to_mask(
    swt: NDArray[np.float32],
    min_stroke: int = 3,
    max_stroke: int = 40,
) -> NDArray[np.uint8]:
    """
    Convert a SWT map to a binary text mask by thresholding stroke widths.

    Args:
        swt: Float SWT map (output of stroke_width_transform).
        min_stroke: Minimum allowed stroke width (inclusive).
        max_stroke: Maximum allowed stroke width (inclusive).

    Returns:
        Binary mask (uint8, 0/255) where text regions are white.
    """
    mask = np.where((swt >= min_stroke) & (swt <= max_stroke), 255, 0).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Connected component grouping (for text candidate extraction)
# ---------------------------------------------------------------------------

def group_swt_components(
    swt_map: NDArray[np.float32],
    min_stroke: int = 3,
    max_stroke: int = 40,
    variance_threshold: float = 2.0,
    area_min: int = 30,
) -> list[tuple[int, int, int, int]]:
    """
    Extract candidate text regions by grouping connected pixels with
    similar stroke width.

    Steps:
    1. Threshold SWT map.
    2. Find connected components.
    3. Filter by stroke width variance and component area.

    Args:
        swt_map: SWT map from stroke_width_transform.
        min_stroke: Minimum stroke width to consider.
        max_stroke: Maximum stroke width to consider.
        variance_threshold: Max allowed stroke width standard deviation
            within a component (relative to its mean).
        area_min: Minimum component area in pixels.

    Returns:
        List of bounding boxes (x, y, w, h) for valid text candidates.
    """
    mask = swt_to_mask(swt_map, min_stroke, max_stroke)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    boxes = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_min:
            continue

        # Compute stroke width variance within the component
        component_mask = (labels == i).astype(np.uint8)
        swt_vals = swt_map[component_mask > 0]
        swt_vals = swt_vals[swt_vals > 0]  # ignore unassigned
        if len(swt_vals) < 5:
            continue

        mean_val = np.mean(swt_vals)
        if mean_val == 0:
            continue
        std_val = np.std(swt_vals)
        if std_val / mean_val > variance_threshold:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, w, h))

    return boxes


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def draw_stroke_width(
    image: NDArray[np.uint8],
    swt_map: NDArray[np.float32],
    max_display_width: int = 40,
) -> NDArray[np.uint8]:
    """
    Create a color-coded visualization of the SWT map overlaid on the image.

    Args:
        image: Original BGR or grayscale image.
        swt_map: SWT float32 map.
        max_display_width: Stroke width mapped to max color (255).

    Returns:
        BGR image with SWT heatmap overlay.
    """
    if image.ndim == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display = image.copy()

    # Normalize SWT to [0, 255]
    swt_clipped = np.clip(swt_map, 0, max_display_width)
    swt_vis = (swt_clipped / max_display_width * 255).astype(np.uint8)
    swt_color = cv2.applyColorMap(swt_vis, cv2.COLORMAP_JET)

    # Blend where SWT > 0
    mask = (swt_map > 0).astype(np.uint8) * 255
    mask_3c = cv2.merge([mask, mask, mask])

    alpha = 0.6
    display = np.where(mask_3c > 0, (alpha * swt_color + (1 - alpha) * display), display).astype(np.uint8)
    return display


def draw_swt_boxes(
    image: NDArray[np.uint8],
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    inplace: bool = False,
) -> NDArray[np.uint8]:
    """
    Draw SWT candidate bounding boxes on an image.

    Args:
        image: Input BGR or grayscale image.
        boxes: List of (x, y, w, h) rectangles.
        color: Rectangle color (BGR).
        thickness: Line thickness.
        inplace: Whether to modify input image.

    Returns:
        Image with drawn boxes.
    """

def detect_text_swt(
    gray: NDArray[np.uint8],
    edge_low: float = 50.0,
    edge_high: float = 150.0,
    max_stroke_width: int = 70,
    min_stroke: int = 3,
    max_stroke: int = 40,
    variance_threshold: float = 2.0,
    area_min: int = 30,
) -> tuple[NDArray[np.uint8], list[tuple[int, int, int, int]], NDArray[np.float32]]:
    """
    Complete text detection pipeline using Stroke Width Transform.
    
    This is a convenience function that wraps the full SWT pipeline:
    1. Compute SWT map
    2. Threshold to get text mask
    3. Extract connected components as text candidates
    
    Args:
        gray: Grayscale input image (uint8)
        edge_low: Canny low threshold
        edge_high: Canny high threshold
        max_stroke_width: Maximum stroke width to search
        min_stroke: Minimum stroke width for text
        max_stroke: Maximum stroke width for text
        variance_threshold: Max allowed stroke width variance in component
        area_min: Minimum component area in pixels
    
    Returns:
        Tuple of (binary_mask, bounding_boxes, swt_map)
    """
    # Compute SWT
    swt_map = stroke_width_transform(
        gray, edge_low, edge_high, max_stroke_width
    )
    
    # Get binary mask
    mask = swt_to_mask(swt_map, min_stroke, max_stroke)
    
    # Get candidate boxes
    boxes = group_swt_components(
        swt_map, min_stroke, max_stroke, variance_threshold, area_min
    )
    
    return mask, boxes, swt_map
    
    out = image if inplace else image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out
