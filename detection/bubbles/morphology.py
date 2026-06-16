"""
Low-level morphological tools for mask processing.

This module provides simple, robust functions for:
- Dilation, erosion, opening, closing
- Noise cleaning (removing small artifacts, filling holes)
- Light smoothing of mask edges (Gaussian blur + threshold)
All functions work with binary masks (uint8, values 0 or 255) and use OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def _ensure_binary_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a mask to strict 0/255 binary format (uint8)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # If values are 0/1, scale to 0/255
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    # Apply a hard threshold just in case
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def _get_kernel(
    kernel_size: int, kernel_shape: int = cv2.MORPH_ELLIPSE
) -> NDArray[np.uint8]:
    """Create a structuring element of the given size and shape."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd positive integer")
    return cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))


def dilate(
    mask: NDArray[np.uint8],
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> NDArray[np.uint8]:
    """
    Apply dilation to a binary mask.

    Args:
        mask: Input binary mask (0/255, uint8).
        kernel_size: Size of the structuring element (odd).
        iterations: Number of times dilation is applied.
        kernel_shape: Shape of the structuring element (e.g., MORPH_ELLIPSE).

    Returns:
        Dilated mask (uint8).
    """
    mask = _ensure_binary_mask(mask)
    kernel = _get_kernel(kernel_size, kernel_shape)
    return cv2.dilate(mask, kernel, iterations=iterations)


def erode(
    mask: NDArray[np.uint8],
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> NDArray[np.uint8]:
    """
    Apply erosion to a binary mask.

    Args:
        mask: Input binary mask (0/255, uint8).
        kernel_size: Size of the structuring element (odd).
        iterations: Number of times erosion is applied.
        kernel_shape: Shape of the structuring element.

    Returns:
        Eroded mask (uint8).
    """
    mask = _ensure_binary_mask(mask)
    kernel = _get_kernel(kernel_size, kernel_shape)
    return cv2.erode(mask, kernel, iterations=iterations)


def open_mask(
    mask: NDArray[np.uint8],
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> NDArray[np.uint8]:
    """
    Morphological opening (erosion followed by dilation).

    Removes small bright spots (noise) while preserving larger regions.

    Args:
        mask: Input binary mask (0/255, uint8).
        kernel_size: Size of the structuring element (odd).
        iterations: Number of times the opening operation is applied.
        kernel_shape: Shape of the structuring element.

    Returns:
        Opened mask (uint8).
    """
    mask = _ensure_binary_mask(mask)
    kernel = _get_kernel(kernel_size, kernel_shape)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def close_mask(
    mask: NDArray[np.uint8],
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> NDArray[np.uint8]:
    """
    Morphological closing (dilation followed by erosion).

    Fills small holes and gaps inside objects.

    Args:
        mask: Input binary mask (0/255, uint8).
        kernel_size: Size of the structuring element (odd).
        iterations: Number of times the closing operation is applied.
        kernel_shape: Shape of the structuring element.

    Returns:
        Closed mask (uint8).
    """
    mask = _ensure_binary_mask(mask)
    kernel = _get_kernel(kernel_size, kernel_shape)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def clean_noise(
    mask: NDArray[np.uint8],
    open_kernel_size: int = 3,
    close_kernel_size: int | None = None,
    iterations: int = 1,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> NDArray[np.uint8]:
    """
    Remove noise by sequential opening (kill small spots) and closing (fill holes).

    Args:
        mask: Input binary mask.
        open_kernel_size: Kernel size for opening.
        close_kernel_size: Kernel size for closing. If None, uses same as open_kernel_size.
        iterations: Number of iterations for each operation.
        kernel_shape: Shape of structuring elements.

    Returns:
        Cleaned mask (uint8).
    """
    if close_kernel_size is None:
        close_kernel_size = open_kernel_size

    mask = _ensure_binary_mask(mask)
    mask = open_mask(mask, open_kernel_size, iterations, kernel_shape)
    mask = close_mask(mask, close_kernel_size, iterations, kernel_shape)
    return mask


def improve_mask(
    mask: NDArray[np.uint8],
    blur_ksize: int = 3,
    blur_sigma: float = 0.0,
    threshold: int = 127,
) -> NDArray[np.uint8]:
    """
    Lightly smooth mask edges with a Gaussian blur, then re-threshold.

    This helps soften jagged edges without changing the overall shape.
    Gaussian blur is intentionally kept light (small kernel).

    Args:
        mask: Input binary mask (0/255, uint8).
        blur_ksize: Gaussian kernel size (odd). Must be odd.
        blur_sigma: Gaussian sigma. 0 means auto-detect from kernel size.
        threshold: Threshold value for binarization (0-255).

    Returns:
        Smoothed mask (uint8).
    """
    if blur_ksize % 2 == 0 or blur_ksize < 1:
        raise ValueError("blur_ksize must be an odd positive integer")

    mask = _ensure_binary_mask(mask)
    blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), blur_sigma)
    _, result = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return result


def apply_morphology(
    mask: NDArray[np.uint8],
    operation: str,
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> NDArray[np.uint8]:
    """
    Generic morphological operation dispatcher.

    Args:
        mask: Input binary mask.
        operation: One of 'dilate', 'erode', 'open', 'close'.
        kernel_size: Kernel size (odd).
        iterations: Number of iterations.
        kernel_shape: Shape of structuring element.

    Returns:
        Processed mask (uint8).

    Raises:
        ValueError: If operation is unsupported.
    """
    op_map = {
        "dilate": dilate,
        "erode": erode,
        "open": open_mask,
        "close": close_mask,
    }
    if operation not in op_map:
        raise ValueError(
            f"Unsupported operation '{operation}'. Choose from {list(op_map.keys())}"
        )
    return op_map[operation](
        mask, kernel_size=kernel_size, iterations=iterations, kernel_shape=kernel_shape
    )