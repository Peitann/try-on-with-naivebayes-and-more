"""
Webcam capture without OpenCV
Uses platform-specific methods or pygame as fallback
"""

import numpy as np
from typing import Optional, Tuple
import platform
import sys


class WebcamCapture:
    """
    Cross-platform webcam capture without OpenCV.
    """
    
    def __init__(self, camera_index: int = 0, size: Tuple[int, int] = (640, 480)):
        """
        Initialize webcam capture.
        
        Args:
            camera_index: Camera device index
            size: Frame size (width, height)
        """
        self.camera_index = camera_index
        self.size = size
        self.camera = None
        self.method = None
        
        # Try to initialize camera
        self._init_camera()
    
    def _init_camera(self):
        """Initialize camera using available method."""
        # Try pygame first (most compatible)
        if self._init_pygame():
            self.method = "pygame"
            print("Using pygame for camera capture")
            return
        
        # Try platform-specific methods
        system = platform.system()
        
        if system == "Windows":
            if self._init_windows():
                self.method = "windows"
                print("Using Windows DirectShow for camera capture")
                return
        
        print("Warning: No camera method available!")
        print("Please install pygame: pip install pygame")
        self.method = None
    
    def _init_pygame(self) -> bool:
        """Initialize pygame camera."""
        try:
            import pygame.camera
            pygame.camera.init()
            
            cameras = pygame.camera.list_cameras()
            if not cameras:
                return False
            
            self.camera = pygame.camera.Camera(cameras[self.camera_index], self.size)
            self.camera.start()
            return True
        except:
            return False
    
    def _init_windows(self) -> bool:
        """Initialize Windows DirectShow camera (requires additional setup)."""
        # This would require win32com or similar
        # For now, return False
        return False
    
    def read(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera.
        
        Returns:
            Frame as numpy array (H, W, 3) or None if capture failed
        """
        if self.method == "pygame":
            return self._read_pygame()
        elif self.method == "windows":
            return self._read_windows()
        else:
            return None
    
    def _read_pygame(self) -> Optional[np.ndarray]:
        """Read frame using pygame."""
        try:
            import pygame.image
            
            if self.camera is None:
                return None
            
            # Capture frame
            image_surface = self.camera.get_image()
            
            # Convert to numpy array
            image_str = pygame.image.tostring(image_surface, 'RGB')
            image = np.frombuffer(image_str, dtype=np.uint8)
            image = image.reshape((self.size[1], self.size[0], 3))
            
            return image
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None
    
    def _read_windows(self) -> Optional[np.ndarray]:
        """Read frame using Windows DirectShow."""
        # Placeholder for Windows-specific implementation
        return None
    
    def release(self):
        """Release camera resources."""
        if self.method == "pygame" and self.camera is not None:
            try:
                self.camera.stop()
            except:
                pass
        
        self.camera = None
    
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.camera is not None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()


# Alternative: Generate dummy frames for testing
class DummyWebcam:
    """
    Dummy webcam that generates test frames.
    Useful for testing without a real camera.
    """
    
    def __init__(self, size: Tuple[int, int] = (640, 480)):
        self.size = size
        self.frame_count = 0
    
    def read(self) -> np.ndarray:
        """Generate a dummy frame."""
        w, h = self.size
        
        # Create gradient image
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add some color variation
        frame[:, :, 0] = (np.arange(w) * 255 // w).astype(np.uint8)  # Red gradient
        frame[:, :, 1] = (np.arange(h)[:, np.newaxis] * 255 // h).astype(np.uint8)  # Green gradient
        frame[:, :, 2] = 128  # Blue constant
        
        # Add moving circle (simulated face)
        center_x = w // 2 + int(50 * np.sin(self.frame_count * 0.05))
        center_y = h // 2 + int(30 * np.cos(self.frame_count * 0.05))
        
        # Draw circle manually
        for y in range(max(0, center_y - 60), min(h, center_y + 60)):
            for x in range(max(0, center_x - 60), min(w, center_x + 60)):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 60:
                    # Skin-like color
                    frame[y, x] = [220, 180, 140]
        
        self.frame_count += 1
        return frame
    
    def release(self):
        """Release (no-op for dummy)."""
        pass
    
    def is_opened(self) -> bool:
        """Always returns True."""
        return True


if __name__ == "__main__":
    print("Testing webcam capture...")
    
    # Try real camera
    cam = WebcamCapture()
    
    if cam.is_opened():
        print("Camera initialized successfully!")
        
        # Capture a few frames
        for i in range(5):
            frame = cam.read()
            if frame is not None:
                print(f"Frame {i+1}: shape={frame.shape}, dtype={frame.dtype}")
            else:
                print(f"Frame {i+1}: Failed to capture")
        
        cam.release()
    else:
        print("Camera initialization failed!")
        print("Falling back to dummy camera...")
        
        dummy = DummyWebcam()
        frame = dummy.read()
        print(f"Dummy frame: shape={frame.shape}, dtype={frame.dtype}")
