"""
Unified mask processing – merging, prioritisation, and cleanup.

This module combines binary masks from different detectors (e.g., text, bubbles)
into a single coherent representation. Overlapping regions are resolved using
a user-defined priority order. After merging, morphological cleaning removes
small islands and holes, producing a clean final mask.

Typical usage:
    from masks import merge_labeled, final_cleanup_binary

    masks = {"text": text_mask, "bubbles": bubbles_mask}
    # Text is more important than bubbles
    labeled = merge_labeled(masks, priority=["text", "bubbles"])
    # Convert to a simple binary mask and clean
    binary = (labeled > 0).astype(np.uint8) * 255
    clean  = final_cleanup_binary(binary)
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_shape_match(
    masks: dict[str, NDArray[np.uint8]],
    expected_shape: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Verify all masks have the same shape. Return the common shape."""
    if not masks:
        raise ValueError("At least one mask is required.")

    shapes = {name: arr.shape for name, arr in masks.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        raise ValueError(f"Masks have different shapes: {shapes}")
    shape = next(iter(unique_shapes))
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"Masks shape {shape} does not match expected {expected_shape}")
    return shape


def _to_binary(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Force a mask to 0/255 binary format."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


# ---------------------------------------------------------------------------
# Merging with priority
# ---------------------------------------------------------------------------

def merge_labeled(
    masks: dict[str, NDArray[np.uint8]],
    priority: list[str],
) -> NDArray[np.uint8]:
    """
    Merge multiple binary masks into a single labelled mask.

    Each pixel receives the label of the highest‑priority mask that covers it.
    Labels are 1‑based indices according to `priority` (1 = highest priority).

    Args:
        masks: Dictionary mapping detector names to their binary masks
               (uint8, values 0 or 255).
        priority: Ordered list of detector names. The first name has the
                  highest priority and will overwrite all others in
                  overlapping regions.

    Returns:
        Labelled mask (uint8) with values:
            - 0  : background (no detector claims the pixel)
            - 1+ : foreground, labelled according to priority list.
                   A pixel belonging to the highest‑priority detector among
                   those that cover it.

    Raises:
        ValueError: If `masks` is empty, shapes differ, or a priority name
                    is missing from `masks`.
    """
    if not masks:
        raise ValueError("masks dictionary must not be empty.")
    if not priority:
        raise ValueError("priority list must not be empty.")

    # Ensure all masks exist for the given priorities
    for name in priority:
        if name not in masks:
            raise ValueError(f"Priority name '{name}' not found in masks keys: {list(masks.keys())}")

    # Validate shape consistency
    shape = _ensure_shape_match(masks)

    # Convert all to strict binary
    masks_bin = {name: _to_binary(mask) for name, mask in masks.items()}

    # Build labelled map (start with background = 0)
    labeled = np.zeros(shape, dtype=np.uint8)

    # Assign labels in priority order (highest first, so it stays if overlap)
    # We iterate from highest to lowest: for each pixel, set label if
    # mask is foreground and pixel not yet claimed.
    for label_idx, name in enumerate(priority, start=1):
        mask = masks_bin[name]
        # Where mask is 255 AND labeled is still 0 → assign current label
        labeled = np.where(
            (mask == 255) & (labeled == 0),
            label_idx,
            labeled,
        )

    return labeled


def merge_binary(
    masks: dict[str, NDArray[np.uint8]],
    priority: list[str],
) -> NDArray[np.uint8]:
    """
    Merge masks into a single binary foreground mask (0/255).

    This is essentially the union of all input masks, but if you intend to
    apply priority rules (e.g., text overwrites bubbles) you should use
    `merge_labeled` and then threshold the result. This function simply
    returns the logical OR of all masks.

    Args:
        masks: Dictionary of binary masks.
        priority: Ignored; kept for API symmetry with `merge_labeled`.

    Returns:
        Binary mask (uint8, 0/255) where any foreground is 255.
    """
    _ = priority  # not used here, but kept for a uniform interface
    if not masks:
        raise ValueError("masks dictionary must not be empty.")

    shape = _ensure_shape_match(masks)
    masks_bin = {name: _to_binary(mask) for name, mask in masks.items()}

    combined = np.zeros(shape, dtype=np.uint8)
    for mask in masks_bin.values():
        combined = cv2.bitwise_or(combined, mask)

    return combined


# ---------------------------------------------------------------------------
# Morphological cleanup
# ---------------------------------------------------------------------------

def cleanup_mask(
    mask: NDArray[np.uint8],
    open_kernel_size: int = 3,
    close_kernel_size: int | None = None,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> NDArray[np.uint8]:
    """
    Remove small noise and fill tiny holes in a binary mask.

    Applies morphological opening (erosion → dilation) followed by
    closing (dilation → erosion). This is a light, fast post‑processing
    step that does not alter the overall shape of large regions.

    Args:
        mask: Binary input mask (0/255, uint8).
        open_kernel_size: Kernel size for opening (odd integer).
        close_kernel_size: Kernel size for closing (odd integer).
            If None, the same as `open_kernel_size` is used.
        kernel_shape: Structuring element shape (e.g., cv2.MORPH_ELLIPSE).

    Returns:
        Cleaned binary mask (uint8).
    """
    mask = _to_binary(mask)
    if close_kernel_size is None:
        close_kernel_size = open_kernel_size

    kernel_open = cv2.getStructuringElement(kernel_shape, (open_kernel_size, open_kernel_size))
    kernel_close = cv2.getStructuringElement(kernel_shape, (close_kernel_size, close_kernel_size))

    # Opening: remove small bright spots
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    # Closing: fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    return closed


# ---------------------------------------------------------------------------
# High‑level convenience
# ---------------------------------------------------------------------------

def process_masks(
    masks: dict[str, NDArray[np.uint8]],
    priority: list[str],
    cleanup: bool = True,
    open_kernel_size: int = 3,
    close_kernel_size: int | None = None,
    return_labeled: bool = False,
) -> NDArray[np.uint8]:
    """
    Merge multiple detector masks with priority and optionally clean.

    Args:
        masks: Dictionary of binary masks (name → uint8 mask).
        priority: List of detector names from highest to lowest priority.
        cleanup: Whether to apply morphological cleanup after merging.
        open_kernel_size: Opening kernel size (if cleanup=True).
        close_kernel_size: Closing kernel size (if cleanup=True).
        return_labeled: If True, return the labelled mask (1,2,…).
            If False (default), return a binary mask (0/255) where
            all foreground classes are merged into 255.

    Returns:
        If `return_labeled` is True:
            Labelled mask (uint8, 0 = background, 1+ = class index).
        If `return_labeled` is False:
            Binary mask (uint8, 0/255) after merging and cleaning.
    """
    labeled = merge_labeled(masks, priority)
    if return_labeled:
        # No binarisation – labelled mask cannot be cleaned with binary operators,
        # but a light opening/closing could be applied per class if needed.
        # For simplicity we return as-is.
        return labeled

    # Convert to binary
    binary = np.where(labeled > 0, 255, 0).astype(np.uint8)

    if cleanup:
        binary = cleanup_mask(
            binary,
            open_kernel_size=open_kernel_size,
            close_kernel_size=close_kernel_size,
        )

    return binary