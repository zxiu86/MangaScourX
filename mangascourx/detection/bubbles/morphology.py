"""
Low-level morphological tools for mask processing.
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


def _get_kernel(kernel_size, kernel_shape=cv2.MORPH_ELLIPSE):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd positive integer")
    return cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))


def dilate(mask, kernel_size=3, iterations=1, kernel_shape=cv2.MORPH_ELLIPSE):
    mask = _ensure_binary_mask(mask)
    return cv2.dilate(mask, _get_kernel(kernel_size, kernel_shape), iterations=iterations)


def erode(mask, kernel_size=3, iterations=1, kernel_shape=cv2.MORPH_ELLIPSE):
    mask = _ensure_binary_mask(mask)
    return cv2.erode(mask, _get_kernel(kernel_size, kernel_shape), iterations=iterations)


def open_mask(mask, kernel_size=3, iterations=1, kernel_shape=cv2.MORPH_ELLIPSE):
    mask = _ensure_binary_mask(mask)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _get_kernel(kernel_size, kernel_shape), iterations=iterations)


def close_mask(mask, kernel_size=3, iterations=1, kernel_shape=cv2.MORPH_ELLIPSE):
    mask = _ensure_binary_mask(mask)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _get_kernel(kernel_size, kernel_shape), iterations=iterations)


def clean_noise(mask, open_kernel_size=3, close_kernel_size=None, iterations=1, kernel_shape=cv2.MORPH_ELLIPSE):
    if close_kernel_size is None:
        close_kernel_size = open_kernel_size
    mask = _ensure_binary_mask(mask)
    mask = open_mask(mask, open_kernel_size, iterations, kernel_shape)
    mask = close_mask(mask, close_kernel_size, iterations, kernel_shape)
    return mask


def improve_mask(mask, blur_ksize=3, blur_sigma=0.0, threshold=127):
    if blur_ksize % 2 == 0 or blur_ksize < 1:
        raise ValueError("blur_ksize must be an odd positive integer")
    mask = _ensure_binary_mask(mask)
    blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), blur_sigma)
    _, result = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return result
