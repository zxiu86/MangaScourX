"""
Base contract for all detectors in the pipeline.

Every detector **must** subclass `BaseDetector` and implement the
`detect` method, which takes an image and returns a binary mask
(uint8, values 0 and 255).

This ensures a uniform interface across:
- Morphology‑based detectors (bubbles)
- Text detectors (MSER, SWT, CRAFT)
- Any future plug‑in detectors
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import cv2
import numpy as np
from numpy.typing import NDArray


class BaseDetector(abc.ABC):
    """
    Abstract base class for all mask‑producing detectors.

    Subclasses must override `detect` and may optionally override
    `preprocess` and `postprocess` to standardise the pipeline.

    The contract:
        - `detect` receives an image (BGR or grayscale) and **must**
          return a binary mask (uint8, 0/255).
        - Any extra metadata (bounding boxes, scores) **may** be returned
          in a dictionary via the `detect_with_meta` method, but the
          core `detect` returns only the mask.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Store arbitrary configuration; subclasses can use or ignore."""
        self.config: Dict[str, Any] = kwargs

    @abc.abstractmethod
    def detect(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Run detection and return a binary mask.

        Args:
            image: Input image. Can be BGR (H,W,3) or grayscale (H,W).

        Returns:
            Binary mask of shape (H,W), dtype uint8, with values 0 or 255.
        """
        ...

    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Optional common preprocessing (e.g., convert to grayscale).

        Subclasses may override this to add their own preprocessing,
        but must still return an image (not necessarily grayscale).
        """
        return image

    def postprocess(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Ensure the output mask is a strict 0/255 binary mask.

        Subclasses can override to apply additional cleanup,
        but the base version guarantees the binary format.
        """
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        # If values are 0/1, scale to 0/255
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        # Threshold to force binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def detect_with_meta(self, image: NDArray[np.uint8]) -> Dict[str, Any]:
        """
        Detect and return both the mask and optional metadata.

        By default this calls `detect` and wraps it in a dict.
        Subclasses may override to include bounding boxes, scores, etc.

        Returns:
            Dictionary with at least a 'mask' key.
        """
        mask = self.detect(image)
        return {"mask": mask}


def validate_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Helper that validates an image and converts it to a BGR uint8 array
    if it is grayscale. This can be used by detectors that expect BGR.
    """
    if image.dtype != np.uint8:
        raise TypeError(f"Image dtype must be uint8, got {image.dtype}")
    if image.ndim == 2:
        # Grayscale → convert to BGR for consistency
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        # Single channel 3D → squeeze and convert
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unexpected image shape: {image.shape}")