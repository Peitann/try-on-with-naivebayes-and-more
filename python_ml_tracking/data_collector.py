"""
Data collection module for capturing training images
Uses Pillow's ImageGrab for webcam capture without OpenCV
"""

import numpy as np
from PIL import Image, ImageGrab
import time
import os
from typing import Optional
import platform


class DataCollector:
    """
    Collects training images for skin detection model.
    Uses platform-specific webcam capture without OpenCV.
    """
    
    def __init__(self, output_dir: str = "datasets/training_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.image_count = 0
    
    def capture_from_screen(self, bbox: Optional[tuple] = None) -> np.ndarray:
        """
        Capture screenshot area (for testing purposes).
        
        Args:
            bbox: Optional bounding box (x1, y1, x2, y2)
        
        Returns:
            Image as numpy array
        """
        screenshot = ImageGrab.grab(bbox=bbox)
        return np.array(screenshot)
    
    def save_image(self, image: np.ndarray, prefix: str = "train") -> str:
        """
        Save captured image to disk.
        
        Args:
            image: Image as numpy array
            prefix: Filename prefix
        
        Returns:
            Path to saved image
        """
        filename = f"{prefix}_{self.image_count:04d}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to PIL and save
        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_image.save(filepath)
        
        self.image_count += 1
        return filepath
    
    def capture_sequence(self, num_images: int = 20, interval: float = 1.0, bbox: Optional[tuple] = None):
        """
        Capture a sequence of images with time intervals.
        
        Args:
            num_images: Number of images to capture
            interval: Time interval between captures (seconds)
            bbox: Optional screen region to capture
        """
        print(f"Starting capture sequence: {num_images} images...")
        print(f"Images will be saved to: {self.output_dir}")
        print("Position your face in the webcam view!")
        
        time.sleep(3)  # Give user time to prepare
        
        for i in range(num_images):
            print(f"Capturing image {i+1}/{num_images}...")
            
            try:
                image = self.capture_from_screen(bbox)
                filepath = self.save_image(image, prefix="train")
                print(f"  Saved: {filepath}")
            except Exception as e:
                print(f"  Error capturing image: {e}")
            
            if i < num_images - 1:
                time.sleep(interval)
        
        print(f"\nCapture complete! {self.image_count} images saved.")
        print("Next step: Use the labeling tool to mark skin regions.")


# Note: For actual webcam capture, you'll need to implement platform-specific code
# Windows: Use win32 API or ctypes
# Linux: Use v4l2 bindings
# macOS: Use AVFoundation bindings
# 
# Alternative: Use pygame.camera (pure Python, no OpenCV)

def capture_with_pygame_camera(camera_index: int = 0, size: tuple = (640, 480)) -> Optional[np.ndarray]:
    """
    Capture image using pygame.camera as OpenCV alternative.
    
    Args:
        camera_index: Camera device index
        size: Image size (width, height)
    
    Returns:
        Image as numpy array or None if pygame not available
    """
    try:
        import pygame.camera
        import pygame.image
        
        # Initialize camera
        pygame.camera.init()
        cameras = pygame.camera.list_cameras()
        
        if not cameras:
            print("No cameras detected")
            return None
        
        camera = pygame.camera.Camera(cameras[camera_index], size)
        camera.start()
        
        # Capture frame
        image_surface = camera.get_image()
        
        # Convert to numpy array
        image_str = pygame.image.tostring(image_surface, 'RGB')
        image = np.frombuffer(image_str, dtype=np.uint8)
        image = image.reshape((size[1], size[0], 3))
        
        camera.stop()
        
        return image
    
    except ImportError:
        print("pygame not available. Install with: pip install pygame")
        return None
    except Exception as e:
        print(f"Error capturing with pygame: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    print("=" * 50)
    print("Training Data Collection")
    print("=" * 50)
    print("\nThis will capture images for training the skin detection model.")
    print("\nOptions:")
    print("1. Screen capture (for testing)")
    print("2. Pygame camera (requires pygame)")
    print("3. Manual - add images to datasets/training_images manually")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        # Capture screen region
        print("\nPosition a photo/video of your face on screen")
        input("Press Enter when ready...")
        collector.capture_sequence(num_images=20, interval=1.0)
    
    elif choice == "2":
        # Try pygame camera
        print("\nStarting camera capture...")
        for i in range(20):
            image = capture_with_pygame_camera()
            if image is not None:
                collector.save_image(image)
                print(f"Captured image {i+1}/20")
                time.sleep(1)
            else:
                print("Camera capture failed")
                break
    
    else:
        print("\nManual mode:")
        print("1. Add 20-100 face images to: datasets/training_images/")
        print("2. Images should show your face in various lighting conditions")
        print("3. Then run the labeling tool to mark skin regions")
