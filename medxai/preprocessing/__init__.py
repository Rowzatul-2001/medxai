from typing import Tuple

import cv2
import numpy as np


def clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Adaptive Histogram Equalization (CLAHE) to an image.

    Handles multi-channel images by converting to LAB color space and
    applying CLAHE to the L channel only.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D grayscale or 3D multi-channel).
    clip_limit : float, optional
        Threshold for contrast limiting, by default 2.0.
    tile_size : Tuple[int, int], optional
        Size of the grid for CLAHE, by default (8, 8).

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image with same shape as input.

    Raises
    ------
    TypeError
        If input is not a NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array")

    if len(image.shape) == 3:
        # Convert to LAB, apply CLAHE to L channel, convert back
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        cl = clahe_obj.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe_obj.apply(image)


def normalize(
    image: np.ndarray, min_val: float = 0.0, max_val: float = 1.0
) -> np.ndarray:
    """
    Applies Min-Max normalization to an image.

    Safeguards against zero variance by returning zeros when all pixels
    have the same value.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    min_val : float, optional
        Minimum value of output range, by default 0.0.
    max_val : float, optional
        Maximum value of output range, by default 1.0.

    Returns
    -------
    np.ndarray
        Normalized image scaled to [min_val, max_val].
    """
    img_float = image.astype(np.float32)
    img_min, img_max = img_float.min(), img_float.max()
    if img_max == img_min:
        return np.zeros_like(img_float)
    return (img_float - img_min) / (img_max - img_min) * (max_val - min_val) + min_val


def resize(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Resizes an image to the specified dimensions.

    Uses anti-aliasing interpolation (INTER_LINEAR) for high fidelity.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    size : Tuple[int, int], optional
        Target (width, height), by default (256, 256).

    Returns
    -------
    np.ndarray
        Resized image.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def roi_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crops the region of interest based on mask bounding boxes.

    Parameters
    ----------
    image : np.ndarray
        Input image to crop.
    mask : np.ndarray
        Binary mask defining the region of interest.

    Returns
    -------
    np.ndarray
        Cropped image containing only the masked region.
        Returns original image if mask is empty.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return image  # Return original if mask is empty
    y_min, x_min = coords.min(axis=0)[:2]
    y_max, x_max = coords.max(axis=0)[:2]
    return image[y_min : y_max + 1, x_min : x_max + 1]


def crop_roi(image: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Crops an image using a bounding box.

    Parameters
    ----------
    image : np.ndarray
        Input image to crop.
    bbox : Tuple[float, float, float, float]
        Bounding box as (ymin, xmin, ymax, xmax).

    Returns
    -------
    np.ndarray
        Cropped image.

    Raises
    ------
    ValueError
        If bbox does not contain exactly 4 values, or if crop region
        is empty after clipping to image bounds.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain exactly 4 values, got {len(bbox)}")

    ymin, xmin, ymax, xmax = (int(round(coord)) for coord in bbox)

    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    height, width = image.shape[:2]
    ymin, ymax = max(0, ymin), min(height, ymax)
    xmin, xmax = max(0, xmin), min(width, xmax)

    if ymin == ymax or xmin == xmax:
        raise ValueError(
            f"Crop region is empty after clipping: y=({ymin}, {ymax}), x=({xmin}, {xmax})"
        )

    return image[ymin:ymax, xmin:xmax]


def apply_ct_window(
    image: np.ndarray, window_center: float, window_width: float
) -> np.ndarray:
    """
    Applies CT windowing (level/width) to an image.

    Clips pixel values to the window range and normalizes to [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input CT image.
    window_center : float
        Center of the display window (level).
    window_width : float
        Width of the display window.

    Returns
    -------
    np.ndarray
        Windowed image normalized to [0, 1].

    Raises
    ------
    TypeError
        If input is not a NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    min_val = window_center - (window_width / 2.0)
    max_val = window_center + (window_width / 2.0)

    clipped = np.clip(image, min_val, max_val)
    normalized = (clipped - min_val) / (max_val - min_val)
    return normalized
