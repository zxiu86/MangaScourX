"""
Contour detection and processing tools for bubble masks.
"""
from __future__ import annotations
import cv2
import numpy as np
from numpy.typing import NDArray


def _ensure_binary_mask(mask):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def find_contours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    mask = _ensure_binary_mask(mask)
    contours, hierarchy = cv2.findContours(mask.copy(), mode, method)
    return contours, hierarchy


def filter_contours_by_area(contours, min_area=0.0, max_area=float("inf")):
    if min_area == 0.0 and max_area == float("inf"):
        return contours
    return [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]


def get_convex_hulls(contours):
    return [cv2.convexHull(c) for c in contours if cv2.convexHull(c) is not None]


def get_bounding_rects(contours):
    return [cv2.boundingRect(c) for c in contours]


def get_bounding_circles(contours):
    circles = []
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        circles.append(((x, y), radius))
    return circles


def detect_bubbles(mask, min_area=0.0, max_area=float("inf")):
    mask = _ensure_binary_mask(mask)
    contours, _ = find_contours(mask, mode=cv2.RETR_EXTERNAL)
    contours = filter_contours_by_area(contours, min_area, max_area)
    return {
        "contours": contours,
        "hulls": get_convex_hulls(contours),
        "bounding_rects": get_bounding_rects(contours),
        "bounding_circles": get_bounding_circles(contours),
    }
