"""
Image processing utilities without OpenCV
Implements manual RGB to HSV conversion and basic image operations
"""

import numpy as np
from typing import Tuple


def rgb_to_hsv(rgb_image: np.ndarray) -> np.ndarray:
    """
    Manually convert RGB image to HSV color space.
    
    Args:
        rgb_image: numpy array of shape (H, W, 3) with values in [0, 255]
    
    Returns:
        HSV image as numpy array with shape (H, W, 3)
        H: [0, 360], S: [0, 100], V: [0, 100]
    """
    # Normalize RGB values to [0, 1]
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    
    r = rgb_normalized[:, :, 0]
    g = rgb_normalized[:, :, 1]
    b = rgb_normalized[:, :, 2]
    
    # Find max and min channel values
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    
    # Initialize HSV arrays
    h = np.zeros_like(max_c)
    s = np.zeros_like(max_c)
    v = max_c
    
    # Calculate Hue
    # Where delta is not zero
    mask_delta = delta != 0
    
    # Red is max
    mask_r = (max_c == r) & mask_delta
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    
    # Green is max
    mask_g = (max_c == g) & mask_delta
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    
    # Blue is max
    mask_b = (max_c == b) & mask_delta
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    
    # Calculate Saturation
    mask_v = max_c != 0
    s[mask_v] = delta[mask_v] / max_c[mask_v]
    
    # Convert to standard ranges: H: [0, 360], S: [0, 100], V: [0, 100]
    hsv_image = np.stack([h, s * 100, v * 100], axis=-1)
    
    return hsv_image


def create_binary_mask(condition_array: np.ndarray) -> np.ndarray:
    """
    Create a binary mask from boolean condition array.
    
    Args:
        condition_array: Boolean numpy array
    
    Returns:
        Binary mask (0 or 255)
    """
    return (condition_array * 255).astype(np.uint8)


def get_skin_mask_from_predictions(predictions: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert ML predictions to binary mask image.
    
    Args:
        predictions: 1D array of predictions (1 for skin, 0 for non-skin)
        shape: Original image shape (height, width)
    
    Returns:
        Binary mask as 2D array
    """
    mask = predictions.reshape(shape)
    return (mask * 255).astype(np.uint8)


def morphological_opening(binary_mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Manual morphological opening (erosion followed by dilation) to remove noise.
    
    Args:
        binary_mask: Binary image (0 or 255)
        kernel_size: Size of structuring element
    
    Returns:
        Cleaned binary mask
    """
    # Simple erosion
    eroded = erosion(binary_mask, kernel_size)
    # Simple dilation
    opened = dilation(eroded, kernel_size)
    return opened


def erosion(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Manual erosion operation."""
    h, w = binary_mask.shape
    result = np.zeros_like(binary_mask)
    pad = kernel_size // 2
    
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            region = binary_mask[i-pad:i+pad+1, j-pad:j+pad+1]
            if np.all(region == 255):
                result[i, j] = 255
    
    return result


def dilation(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Manual dilation operation."""
    h, w = binary_mask.shape
    result = np.zeros_like(binary_mask)
    pad = kernel_size // 2
    
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            region = binary_mask[i-pad:i+pad+1, j-pad:j+pad+1]
            if np.any(region == 255):
                result[i, j] = 255
    
    return result
