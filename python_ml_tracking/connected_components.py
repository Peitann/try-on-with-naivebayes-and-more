"""
Manual connected component analysis without OpenCV
Used to find the largest skin region (the face)
"""

import numpy as np
from typing import Tuple, List


class ConnectedComponents:
    """
    Manual implementation of connected component labeling.
    Uses flood-fill algorithm to identify connected regions.
    """
    
    def __init__(self):
        self.labels = None
        self.num_components = 0
        self.component_sizes = []
        self.component_bboxes = []
    
    def find_components(self, binary_mask: np.ndarray) -> int:
        """
        Find all connected components in binary mask.
        
        Args:
            binary_mask: Binary image (0 or 255)
        
        Returns:
            Number of components found
        """
        h, w = binary_mask.shape
        self.labels = np.zeros((h, w), dtype=np.int32)
        self.num_components = 0
        self.component_sizes = []
        self.component_bboxes = []
        
        # Convert to binary (0 or 1)
        binary = (binary_mask > 127).astype(np.uint8)
        
        # Scan image and label components
        for i in range(h):
            for j in range(w):
                if binary[i, j] == 1 and self.labels[i, j] == 0:
                    # Found new component, perform flood fill
                    self.num_components += 1
                    size, bbox = self._flood_fill(binary, i, j, self.num_components)
                    self.component_sizes.append(size)
                    self.component_bboxes.append(bbox)
        
        return self.num_components
    
    def _flood_fill(self, binary: np.ndarray, start_y: int, start_x: int, label: int) -> Tuple[int, Tuple[int, int, int, int]]:
        """
        Flood fill algorithm to label connected pixels.
        
        Args:
            binary: Binary array (0 or 1)
            start_y: Starting y coordinate
            start_x: Starting x coordinate
            label: Label to assign to this component
        
        Returns:
            Tuple of (component_size, bounding_box)
            bounding_box: (x1, y1, x2, y2)
        """
        h, w = binary.shape
        stack = [(start_y, start_x)]
        size = 0
        
        min_x, min_y = start_x, start_y
        max_x, max_y = start_x, start_y
        
        while stack:
            y, x = stack.pop()
            
            # Check bounds
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            
            # Check if already labeled or not foreground
            if binary[y, x] == 0 or self.labels[y, x] != 0:
                continue
            
            # Label this pixel
            self.labels[y, x] = label
            size += 1
            
            # Update bounding box
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            
            # Add 4-connected neighbors to stack
            stack.append((y-1, x))    # Up
            stack.append((y+1, x))    # Down
            stack.append((y, x-1))    # Left
            stack.append((y, x+1))    # Right
        
        return size, (min_x, min_y, max_x, max_y)
    
    def get_largest_component(self) -> Tuple[int, Tuple[int, int, int, int]]:
        """
        Get the largest connected component and its bounding box.
        
        Returns:
            Tuple of (component_label, bounding_box)
            Returns (0, (0, 0, 0, 0)) if no components found
        """
        if self.num_components == 0:
            return 0, (0, 0, 0, 0)
        
        # Find component with maximum size
        max_idx = np.argmax(self.component_sizes)
        largest_label = max_idx + 1  # Labels start from 1
        largest_bbox = self.component_bboxes[max_idx]
        
        return largest_label, largest_bbox
    
    def get_component_mask(self, label: int) -> np.ndarray:
        """
        Get binary mask for a specific component.
        
        Args:
            label: Component label
        
        Returns:
            Binary mask (0 or 255)
        """
        mask = (self.labels == label).astype(np.uint8) * 255
        return mask


def find_largest_skin_region(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the largest connected skin region and return its bounding box.
    This is assumed to be the face.
    
    Args:
        binary_mask: Binary skin mask (0 or 255)
    
    Returns:
        Bounding box as (x1, y1, x2, y2)
        Returns (0, 0, 0, 0) if no region found
    """
    cc = ConnectedComponents()
    num_components = cc.find_components(binary_mask)
    
    if num_components == 0:
        return (0, 0, 0, 0)
    
    _, bbox = cc.get_largest_component()
    return bbox
