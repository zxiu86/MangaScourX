"""
Central detection orchestrator – decides *what* to use and *when*.

This module receives an image, selects the appropriate detectors
(MSER, CRAFT, bubble segmentation), coordinates their execution,
merges results with configurable priority, and returns a final
unified mask.

Philosophy
----------
- **Simple text**  → MSER (fast, no AI)
- **Complex text** → CRAFT (AI-based, high detail)
- **Bubbles**      → contour‑based segmentation + morphology
- **Fallback**     : if MSER returns too few candidates and CRAFT is
  available, automatically switches to CRAFT.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

# Project imports (adjust if your package structure differs)
from bubbles.morphology import clean_noise
from bubbles.contours import detect_bubbles
from text.mser import detect_text_regions
from text.craft_adapter import CRAFTDetector
from masks import process_masks

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Helper – simple bubble mask generation from an image
# ---------------------------------------------------------------------------

def _generate_bubble_mask(
    image: NDArray[np.uint8],
    *,
    block_size: int = 31,
    c: int = 5,
    morph_open: int = 3,
    morph_close: int = 5,
) -> NDArray[np.uint8]:
    """
    Create a raw bubble candidate mask using adaptive thresholding.

    Bubbles are typically bright, uniform regions on a darker background.
    Adaptive thresholding captures them reliably without manual parameters.

    Args:
        image: BGR or grayscale uint8 image.
        block_size: Adaptive threshold block size (odd).
        c: Constant subtracted from mean.
        morph_open: Kernel size for post‑threshold opening (noise removal).
        morph_close: Kernel size for post‑threshold closing (hole filling).

    Returns:
        Binary mask (uint8, 0/255).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Adaptive threshold to extract bright regions (bubbles)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c,
    )

    # Morphological cleanup
    mask = clean_noise(
        mask,
        open_kernel_size=morph_open,
        close_kernel_size=morph_close,
    )
    return mask


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class Detector:
    """
    Unified detector for text and bubbles.

    Parameters
    ----------
    craft_model_path : str or None
        Path to CRAFT weights. If None, AI detection is disabled.
    device : str
        PyTorch device for CRAFT (e.g., 'cpu', 'cuda').
    mser_params : dict or None
        Keyword arguments for `text.mser.detect_text_regions`.
    craft_params : dict or None
        Keyword arguments forwarded to `CRAFTDetector` (except model_path, device).
    bubble_params : dict or None
        Keyword arguments for `_generate_bubble_mask`.
    merge_priority : list[str]
        Priority order for mask merging (see `masks.process_masks`).
        Default: ['text', 'bubbles'] – text overwrites bubbles.
    final_cleanup : bool
        Apply morphological cleanup to the final merged mask.
    fallback_min_boxes : int
        If MSER detects fewer than this number of text boxes, and CRAFT
        is available, the detector will automatically fallback to CRAFT.

    Example
    -------
    >>> det = Detector(craft_model_path="weights/craft.pth")
    >>> result = det.detect(image_bgr)
    >>> final_mask = result["mask"]
    """

    def __init__(
        self,
        craft_model_path: Optional[str] = None,
        device: str = "cpu",
        mser_params: Optional[dict] = None,
        craft_params: Optional[dict] = None,
        bubble_params: Optional[dict] = None,
        merge_priority: Optional[List[str]] = None,
        final_cleanup: bool = True,
        fallback_min_boxes: int = 3,
    ) -> None:
        # Store parameters
        self.craft_model_path = craft_model_path
        self.device = device
        self.mser_params = mser_params or {}
        self.craft_params = craft_params or {}
        self.bubble_params = bubble_params or {}
        self.merge_priority = merge_priority or ["text", "bubbles"]
        self.final_cleanup = final_cleanup
        self.fallback_min_boxes = fallback_min_boxes

        # Lazy initialisation of CRAFT
        self._craft_detector: Optional[CRAFTDetector] = None
        if craft_model_path:
            try:
                self._craft_detector = CRAFTDetector(
                    craft_model_path, device=device, **self.craft_params
                )
            except Exception as e:
                logger.warning(f"Failed to load CRAFT model: {e}")
                self._craft_detector = None

    # ------------------------------------------------------------------
    # Text detection (decision logic + fallback)
    # ------------------------------------------------------------------

    def _detect_text(
        self, image: NDArray[np.uint8]
    ) -> Tuple[List[Tuple[int, int, int, int]], NDArray[np.uint8]]:
        """
        Run text detection with the appropriate backend.

        Returns
        -------
        boxes : list of (x, y, w, h)
        mask : binary uint8 mask (0/255) drawn from the boxes.
        """
        # 1. Try MSER first (always available)
        boxes = detect_text_regions(image, **self.mser_params)

        # 2. Fallback to CRAFT if MSER under‑performs
        if (
            self._craft_detector is not None
            and len(boxes) < self.fallback_min_boxes
        ):
            logger.info(
                f"MSER found only {len(boxes)} boxes – falling back to CRAFT."
            )
            try:
                boxes = self._craft_detector.detect(image)
            except Exception as e:
                logger.error(f"CRAFT detection failed: {e}")

        # 3. Convert boxes to a binary mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for (x, y, w, h) in boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)

        return boxes, mask

    # ------------------------------------------------------------------
    # Bubble detection
    # ------------------------------------------------------------------

    def _detect_bubbles(
        self, image: NDArray[np.uint8]
    ) -> Tuple[List[Tuple[int, int, int, int]], NDArray[np.uint8]]:
        """
        Run bubble segmentation and return boxes + mask.

        Returns
        -------
        boxes : list of (x, y, w, h) bounding rectangles of bubbles.
        mask : binary uint8 mask (0/255) with bubble regions filled.
        """
        # Generate raw bubble mask
        raw_mask = _generate_bubble_mask(image, **self.bubble_params)

        # Use contour-based detection
        bubble_data = detect_bubbles(raw_mask)
        contours = bubble_data["contours"]
        bounding_rects = bubble_data["bounding_rects"]

        # Draw filled contours to create a clean mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, thickness=-1)

        return bounding_rects, mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: NDArray[np.uint8],
        *,
        enable_text: bool = True,
        enable_bubbles: bool = True,
    ) -> Dict:
        """
        Run detection on an image and return consolidated results.

        Args:
            image: Input image (BGR or grayscale).
            enable_text: Whether to detect text.
            enable_bubbles: Whether to detect bubbles.

        Returns:
            Dictionary containing:
                - 'mask': final unified binary mask (0/255)
                - 'text_boxes': list of (x, y, w, h) (if text enabled)
                - 'bubble_boxes': list of (x, y, w, h) (if bubbles enabled)
                - 'text_mask': text binary mask (if text enabled)
                - 'bubble_mask': bubble binary mask (if bubbles enabled)
                - 'priority': the merge priority used
        """
        masks_dict: Dict[str, NDArray[np.uint8]] = {}
        text_boxes: List = []
        bubble_boxes: List = []

        # 1. Text detection
        if enable_text:
            text_boxes, text_mask = self._detect_text(image)
            masks_dict["text"] = text_mask

        # 2. Bubble detection
        if enable_bubbles:
            bubble_boxes, bubble_mask = self._detect_bubbles(image)
            masks_dict["bubbles"] = bubble_mask

        # 3. Merge with priority
        if len(masks_dict) == 0:
            # Nothing detected → return empty mask
            final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            final_mask = process_masks(
                masks_dict,
                priority=self.merge_priority,
                cleanup=self.final_cleanup,
                return_labeled=False,
            )

        result = {
            "mask": final_mask,
            "priority": self.merge_priority,
        }
        if enable_text:
            result["text_boxes"] = text_boxes
            result["text_mask"] = masks_dict.get("text")
        if enable_bubbles:
            result["bubble_boxes"] = bubble_boxes
            result["bubble_mask"] = masks_dict.get("bubbles")

        return result