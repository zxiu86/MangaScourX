"""
CRAFT adapter for complex text detection in manga and detailed scenes.

This module wraps the CRAFT (Character‑Region Awareness For Text detection)
deep learning model for use within the pipeline. It is intended for high‑detail
images where traditional methods (MSER, SWT) fail to produce reliable
text regions.

Dependencies (will be checked at runtime):
    pip install torch torchvision opencv-python
    # The CRAFT repository must be available as a Python package or
    # the model definition file must be provided.
    # By default we assume the official CRAFT-pytorch code is importable:
    #   from craft import CRAFT
    # If not, the adapter will raise an ImportError with instructions.

Usage:
    from text.craft_adapter import CRAFTDetector

    detector = CRAFTDetector("weights/craft_mlt_25k.pth", device="cpu")
    boxes = detector.detect(image_bgr)
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Optional CRAFT import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    logger.error("PyTorch is not installed. CRAFTDetector will not work.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_craft_net(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Dynamically load the CRAFT model architecture and its weights.

    This function assumes the CRAFT class definition is reachable.
    If the official CRAFT-pytorch repo is not installed, you can provide
    the model definition file path and import it manually.

    Args:
        model_path: Path to the trained .pth file.
        device: 'cpu' or 'cuda'.

    Returns:
        A torch.nn.Module with CRAFT architecture, in evaluation mode.
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required. Install it with: pip install torch torchvision"
        )

    # Try to import CRAFT from the standard source (if installed)
    try:
        from craft import CRAFT
    except ImportError:
        raise ImportError(
            "CRAFT module not found. Please install the CRAFT-pytorch package or "
            "place the CRAFT model definition file on your PYTHONPATH. "
            "You can clone it from https://github.com/clovaai/CRAFT-pytorch"
        )

    net = CRAFT()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    return net


def _preprocess_image(
    image: NDArray[np.uint8],
    canvas_size: int = 1280,
    mag_ratio: float = 1.5,
) -> Tuple[NDArray[np.float32], Tuple[float, float], Tuple[int, int]]:
    """
    Resize image to a fixed canvas while keeping aspect ratio,
    normalise and convert to a torch tensor.

    Args:
        image: BGR or grayscale image.
        canvas_size: Maximum side length after resizing.
        mag_ratio: Magnification ratio (used for small text).

    Returns:
        - preprocessed image as CHW float tensor (1,3,H,W) normalised to [0,1]
        - ratio_w, ratio_h for coordinate mapping
        - original size (height, width)
    """
    if image.ndim == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img = image.copy()

    h, w = img.shape[:2]
    # Resize keeping aspect ratio
    target_size = canvas_size
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = w, h

    # Magnification
    new_w = int(new_w * mag_ratio)
    new_h = int(new_h * mag_ratio)

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Normalise to [0,1] and mean/std normalisation? CRAFT expects pixel values in [0,1]
    # According to the original implementation, they just divide by 255.
    x = img_resized.astype(np.float32) / 255.0
    # BGR -> RGB
    x = x[:, :, ::-1]
    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))
    # Add batch dimension
    x = torch.from_numpy(x).unsqueeze(0)

    ratio_w = w / new_w
    ratio_h = h / new_h

    return x, (ratio_w, ratio_h), (h, w)


def _postprocess_craft_output(
    score_text: NDArray[np.float32],
    score_link: NDArray[np.float32],
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
) -> NDArray[np.uint8]:
    """
    Convert CRAFT output maps to a binary mask using connected components.

    Args:
        score_text: Text region probability map.
        score_link: Link probability map.
        text_threshold: Threshold for text score.
        link_threshold: Threshold for link score.
        low_text: Low bound for text score to consider a pixel.

    Returns:
        Binary mask (uint8, 0/255) of detected text regions.
    """
    # Follow the original CRAFT post-processing
    text_score = score_text.copy()
    link_score = score_link.copy()

    # Use a combined map: text + link
    combined = text_score + link_score
    _, text_mask = cv2.threshold(combined, link_threshold, 1, cv2.THRESH_BINARY)

    # Multiply by text score to suppress low-confidence regions
    text_mask = (text_mask * text_score).astype(np.float32)

    # Binarize the final mask
    _, final_mask = cv2.threshold(text_mask, text_threshold, 1, cv2.THRESH_BINARY)
    return (final_mask * 255).astype(np.uint8)


def _boxes_from_mask(
    binary_mask: NDArray[np.uint8],
    min_area: int = 30,
) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes from a binary mask via connected components.

    Args:
        binary_mask: 0/255 uint8 mask.
        min_area: Minimum component area to keep.

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, w, h))
    return boxes


def _nms_boxes(
    boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """Apply Non‑Maximum Suppression to a list of boxes."""
    if not boxes:
        return []
    box_array = np.array(boxes)
    x1 = box_array[:, 0].astype(float)
    y1 = box_array[:, 1].astype(float)
    x2 = x1 + box_array[:, 2].astype(float)
    y2 = y1 + box_array[:, 3].astype(float)
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return [boxes[i] for i in keep]


# ---------------------------------------------------------------------------
# CRAFTDetector class
# ---------------------------------------------------------------------------

class CRAFTDetector:
    """
    Wrapper around the CRAFT text detection model.

    Provides a simple interface for detecting text regions in complex images
    (e.g., manga, street scenes, fancy typography).

    Example:
        detector = CRAFTDetector("craft_mlt_25k.pth", device="cuda")
        boxes = detector.detect(image)
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        canvas_size: int = 1280,
        mag_ratio: float = 1.5,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        min_area: int = 30,
        nms_threshold: float = 0.5,
    ):
        """
        Args:
            model_path: Path to the CRAFT .pth weights file.
            device: 'cpu' or 'cuda' (or 'cuda:0' etc.).
            canvas_size: Maximum dimension of the resized image.
            mag_ratio: Magnification factor for small text.
            text_threshold: Threshold on the text region score map.
            link_threshold: Threshold on the link (affinity) score map.
            low_text: Low bound used in post-processing.
            min_area: Minimum pixel area for a detected text region.
            nms_threshold: IoU threshold for NMS.
        """
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.min_area = min_area
        self.nms_threshold = nms_threshold

        self.device = torch.device(device)
        self.model = _load_craft_net(model_path, device)
        logger.info(f"CRAFT model loaded on {self.device}")

    def detect(
        self,
        image: NDArray[np.uint8],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect text bounding boxes in an image.

        Args:
            image: Input image (BGR, grayscale, or any OpenCV format).
                   Will be converted to BGR internally if needed.

        Returns:
            List of bounding boxes (x, y, width, height). Coordinates are
            in the original image space.
        """
        # Preprocessing
        x, (ratio_w, ratio_h), (orig_h, orig_w) = _preprocess_image(
            image, self.canvas_size, self.mag_ratio
        )
        x = x.to(self.device)

        # Inference
        with torch.no_grad():
            y, feature = self.model(x)
            # y is (1, 2, H, W) -> score_text, score_link
            score_text = y[0, 0, :, :].cpu().numpy()
            score_link = y[0, 1, :, :].cpu().numpy()

        # Post-process to binary mask (on resized image)
        mask = _postprocess_craft_output(
            score_text,
            score_link,
            self.text_threshold,
            self.link_threshold,
            self.low_text,
        )

        # Resize mask back to original image dimensions
        mask_orig = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Extract bounding boxes
        boxes = _boxes_from_mask(mask_orig, self.min_area)

        # Apply NMS
        boxes = _nms_boxes(boxes, self.nms_threshold)

        return boxes

    def draw_boxes(
        self,
        image: NDArray[np.uint8],
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> NDArray[np.uint8]:
        """
        Utility to draw detected boxes on an image.

        If boxes is None, first runs detection on the image.
        """
        if boxes is None:
            boxes = self.detect(image)
        out = image.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        return out