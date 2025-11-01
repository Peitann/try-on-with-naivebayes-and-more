"""
Interactive labeling tool for marking skin vs non-skin regions
Creates training labels for the ML model
"""

import numpy as np
from PIL import Image, ImageDraw
import os
import json
from typing import List, Tuple, Dict, Optional


class LabelingTool:
    """
    Simple labeling tool for marking skin regions in images.
    Creates labels for training the skin detection classifier.
    """
    
    def __init__(self, image_dir: str = "datasets/training_images", 
                 label_dir: str = "datasets/labels"):
        self.image_dir = image_dir
        self.label_dir = label_dir
        os.makedirs(label_dir, exist_ok=True)
        
        self.current_image = None
        self.current_image_path = None
        self.skin_regions = []
        self.non_skin_regions = []
    
    def load_image(self, image_path: str) -> bool:
        """Load image for labeling."""
        try:
            self.current_image = Image.open(image_path)
            self.current_image_path = image_path
            self.skin_regions = []
            self.non_skin_regions = []
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def add_skin_region(self, bbox: Tuple[int, int, int, int]):
        """
        Add a rectangular region marked as skin.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
        """
        self.skin_regions.append(bbox)
    
    def add_non_skin_region(self, bbox: Tuple[int, int, int, int]):
        """
        Add a rectangular region marked as non-skin.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
        """
        self.non_skin_regions.append(bbox)
    
    def save_labels(self) -> str:
        """
        Save labeled regions to JSON file.
        
        Returns:
            Path to saved label file
        """
        if self.current_image_path is None:
            raise ValueError("No image loaded")
        
        # Create label filename
        image_filename = os.path.basename(self.current_image_path)
        label_filename = os.path.splitext(image_filename)[0] + ".json"
        label_path = os.path.join(self.label_dir, label_filename)
        
        # Prepare label data
        label_data = {
            "image_path": self.current_image_path,
            "image_size": self.current_image.size,
            "skin_regions": self.skin_regions,
            "non_skin_regions": self.non_skin_regions
        }
        
        # Save to JSON
        with open(label_path, 'w') as f:
            json.dump(label_data, f, indent=2)
        
        return label_path
    
    def visualize_labels(self, output_path: Optional[str] = None) -> Image.Image:
        """
        Create visualization of labeled regions.
        
        Args:
            output_path: Optional path to save visualization
        
        Returns:
            PIL Image with visualized labels
        """
        if self.current_image is None:
            raise ValueError("No image loaded")
        
        # Create copy for drawing
        vis_image = self.current_image.copy()
        draw = ImageDraw.Draw(vis_image, 'RGBA')
        
        # Draw skin regions in green
        for bbox in self.skin_regions:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 255, 0, 50))
        
        # Draw non-skin regions in red
        for bbox in self.non_skin_regions:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 50))
        
        if output_path:
            vis_image.save(output_path)
        
        return vis_image


def auto_label_face_region(image_path: str, face_bbox: Tuple[int, int, int, int],
                           background_bboxes: List[Tuple[int, int, int, int]] = None) -> Dict:
    """
    Helper function to automatically create labels for an image.
    
    Args:
        image_path: Path to image
        face_bbox: Bounding box of face region (assumed to be skin)
        background_bboxes: Optional list of background regions (non-skin)
    
    Returns:
        Label data dictionary
    """
    image = Image.open(image_path)
    
    label_data = {
        "image_path": image_path,
        "image_size": image.size,
        "skin_regions": [face_bbox],
        "non_skin_regions": background_bboxes or []
    }
    
    # Save label
    label_dir = "datasets/labels"
    os.makedirs(label_dir, exist_ok=True)
    
    image_filename = os.path.basename(image_path)
    label_filename = os.path.splitext(image_filename)[0] + ".json"
    label_path = os.path.join(label_dir, label_filename)
    
    with open(label_path, 'w') as f:
        json.dump(label_data, f, indent=2)
    
    return label_data


def batch_label_with_simple_heuristic(image_dir: str, output_dir: str = "datasets/labels"):
    """
    Batch label images using simple heuristic (center region as face).
    This is a quick way to create initial labels that can be refined later.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save labels
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Batch labeling {len(image_files)} images...")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        try:
            image = Image.open(img_path)
            w, h = image.size
            
            # Assume face is in center 40% of image
            center_x, center_y = w // 2, h // 2
            face_w, face_h = int(w * 0.4), int(h * 0.4)
            
            face_bbox = (
                center_x - face_w // 2,
                center_y - face_h // 2,
                center_x + face_w // 2,
                center_y + face_h // 2
            )
            
            # Background regions (corners)
            bg_size = 50
            background_bboxes = [
                (0, 0, bg_size, bg_size),  # Top-left
                (w - bg_size, 0, w, bg_size),  # Top-right
                (0, h - bg_size, bg_size, h),  # Bottom-left
                (w - bg_size, h - bg_size, w, h)  # Bottom-right
            ]
            
            # Save label
            auto_label_face_region(img_path, face_bbox, background_bboxes)
            print(f"  Labeled: {img_file}")
        
        except Exception as e:
            print(f"  Error labeling {img_file}: {e}")
    
    print(f"\nLabeling complete! Labels saved to: {output_dir}")
    print("Review and refine labels if needed before training.")


if __name__ == "__main__":
    print("=" * 50)
    print("Image Labeling Tool")
    print("=" * 50)
    print("\nThis tool creates training labels for skin detection.")
    print("\nOptions:")
    print("1. Automatic batch labeling (quick, uses heuristic)")
    print("2. Manual labeling (interactive, more accurate)")
    
    choice = input("\nSelect option (1-2): ").strip()
    
    if choice == "1":
        image_dir = "datasets/training_images"
        if os.path.exists(image_dir):
            batch_label_with_simple_heuristic(image_dir)
        else:
            print(f"Error: Directory not found: {image_dir}")
            print("Please collect training images first.")
    
    elif choice == "2":
        print("\nManual labeling mode:")
        print("For this demo, we'll use automatic labeling.")
        print("For a GUI-based tool, integrate with tkinter or similar.")
        
        # Fall back to automatic
        image_dir = "datasets/training_images"
        if os.path.exists(image_dir):
            batch_label_with_simple_heuristic(image_dir)
        else:
            print(f"Error: Directory not found: {image_dir}")
