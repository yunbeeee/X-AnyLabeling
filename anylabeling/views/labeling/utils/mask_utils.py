import numpy as np
import cv2
from PyQt5.QtCore import QPointF


def apply_brush_to_mask(mask, center_x, center_y, radius, add=True):
    """
    Apply brush to mask at given position
    
    Args:
        mask: numpy array (H, W) - binary mask
        center_x, center_y: brush center coordinates
        radius: brush radius in pixels
        add: True to add pixels, False to remove pixels
    
    Returns:
        modified mask
    """
    if mask is None:
        return mask
        
    h, w = mask.shape
    y, x = np.ogrid[:h, :w]
    
    # Create circular brush
    brush = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    
    # Apply brush
    if add:
        mask = np.logical_or(mask, brush)
    else:
        mask = np.logical_and(mask, ~brush)
    print(f"[DEBUG][apply_brush_to_mask] at ({center_x},{center_y}), add={add}, sum: {mask.sum()}")
    
    return mask.astype(np.uint8)


def mask_to_polygon(mask, simplify=True, tolerance=1.0):
    """
    Convert binary mask to polygon points
    
    Args:
        mask: numpy array (H, W) - binary mask
        simplify: whether to simplify polygon
        tolerance: simplification tolerance
    
    Returns:
        list of polygon points [(x1, y1), (x2, y2), ...]
    """
    if mask is None or mask.sum() == 0:
        return []
    
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return []
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify if requested
    if simplify:
        epsilon = tolerance * cv2.arcLength(largest_contour, True)
        largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of points
    points = [(point[0][0], point[0][1]) for point in largest_contour]
    
    return points


def polygon_to_mask(points, shape):
    """
    Convert polygon points to binary mask
    
    Args:
        points: list of polygon points [(x1, y1), (x2, y2), ...]
        shape: (height, width) of output mask
    
    Returns:
        binary mask as numpy array
    """
    if not points:
        return np.zeros(shape, dtype=np.uint8)
    
    mask = np.zeros(shape, dtype=np.uint8)
    points_array = np.array(points, dtype=np.int32)
    
    cv2.fillPoly(mask, [points_array], (1,))
    
    return mask


def mask_to_rle(mask):
    """
    Convert binary mask to RLE (Run-Length Encoding)
    
    Args:
        mask: numpy array (H, W) - binary mask
    
    Returns:
        RLE string
    """
    if mask is None:
        return ""
    
    # Flatten mask
    flat_mask = mask.flatten()
    
    # Find runs
    runs = []
    current_run = 1
    current_val = flat_mask[0]
    
    for val in flat_mask[1:]:
        if val == current_val:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
            current_val = val
    
    runs.append(current_run)
    
    return " ".join(map(str, runs))


def rle_to_mask(rle_string, shape):
    """
    Convert RLE string to binary mask
    
    Args:
        rle_string: RLE encoded string
        shape: (height, width) of output mask
    
    Returns:
        binary mask as numpy array
    """
    if not rle_string:
        return np.zeros(shape, dtype=np.uint8)
    
    runs = list(map(int, rle_string.split()))
    
    # Reconstruct mask
    flat_mask = []
    current_val = 0
    
    for run_length in runs:
        flat_mask.extend([current_val] * run_length)
        current_val = 1 - current_val  # Toggle between 0 and 1
    
    # Reshape to original shape
    mask = np.array(flat_mask, dtype=np.uint8)
    mask = mask.reshape(shape)
    
    return mask


def mask_to_base64(mask):
    """
    Convert mask to base64 string for storage
    
    Args:
        mask: numpy array - binary mask
    
    Returns:
        base64 encoded string
    """
    import base64
    
    if mask is None:
        return ""
    
    mask_bytes = mask.tobytes()
    return base64.b64encode(mask_bytes).decode('utf-8')


def base64_to_mask(base64_string, shape, dtype=np.uint8):
    """
    Convert base64 string to mask
    
    Args:
        base64_string: base64 encoded string
        shape: (height, width) of output mask
        dtype: data type of mask
    
    Returns:
        mask as numpy array
    """
    import base64
    
    if not base64_string:
        return np.zeros(shape, dtype=dtype)
    
    mask_bytes = base64.b64decode(base64_string)
    mask = np.frombuffer(mask_bytes, dtype=dtype).reshape(shape)
    
    return mask


def resize_mask(mask, new_shape):
    """
    Resize mask to new shape
    
    Args:
        mask: numpy array (H, W) - binary mask
        new_shape: (new_height, new_width)
    
    Returns:
        resized mask
    """
    if mask is None:
        return np.zeros(new_shape, dtype=np.uint8)
    
    resized = cv2.resize(mask.astype(np.uint8), (new_shape[1], new_shape[0]))
    return (resized > 0).astype(np.uint8) 