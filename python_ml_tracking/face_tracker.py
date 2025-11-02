"""
Real-time face tracking using trained ML model
Performs skin detection, connected component analysis, and bounding box extraction
"""

import numpy as np
import pickle
import time
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_ml_tracking.image_utils import rgb_to_hsv, get_skin_mask_from_predictions
from python_ml_tracking.connected_components import find_largest_skin_region
from python_ml_tracking.webcam_capture import WebcamCapture, DummyWebcam


class FaceTracker:
    """
    Real-time face tracking using classical ML skin detection.
    """
    
    def __init__(self, model_path: str, use_dummy_camera: bool = False):
        """
        Initialize face tracker.
        
        Args:
            model_path: Path to trained model pickle file
            use_dummy_camera: Use dummy camera for testing
        """
        # Load trained model
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("Model loaded successfully!")
        
        # Initialize camera
        if use_dummy_camera:
            self.camera = DummyWebcam(size=(640, 480))
        else:
            self.camera = WebcamCapture(camera_index=0, size=(640, 480))
        
        if not self.camera.is_opened():
            raise RuntimeError("Failed to open camera!")
        
        # Tracking state
        self.last_bbox = (0, 0, 0, 0)
        self.smoothing_factor = 0.3  # For bbox smoothing
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Process a single frame and extract face bounding box.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
        
        Returns:
            Tuple of (skin_mask, bounding_box)
            skin_mask: Binary mask of skin regions
            bounding_box: (x1, y1, x2, y2)
        """
        h, w = frame.shape[:2]
        
        # Step 1: Convert RGB to HSV manually
        hsv_image = rgb_to_hsv(frame)
        
        # Step 2: Prepare features for classification
        hsv_flat = hsv_image.reshape(-1, 3)  # Shape: (H*W, 3)
        
        # Step 3: Classify each pixel as skin or non-skin
        predictions = self.model.predict(hsv_flat)  # Shape: (H*W,)
        
        # Step 4: Create binary skin mask
        skin_mask = get_skin_mask_from_predictions(predictions, (h, w))
        
        # Step 5: Find largest connected component (assumed to be face)
        bbox = find_largest_skin_region(skin_mask)
        
        # Step 6: Smooth bounding box (reduce jitter)
        bbox = self._smooth_bbox(bbox)
        
        return skin_mask, bbox
    
    def _smooth_bbox(self, new_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Smooth bounding box using exponential moving average.
        
        Args:
            new_bbox: New bounding box
        
        Returns:
            Smoothed bounding box
        """
        if self.last_bbox == (0, 0, 0, 0):
            self.last_bbox = new_bbox
            return new_bbox
        
        # Check if new bbox is valid
        if new_bbox == (0, 0, 0, 0):
            return self.last_bbox  # Keep last valid bbox
        
        # Smooth each coordinate
        alpha = self.smoothing_factor
        x1 = int(alpha * new_bbox[0] + (1 - alpha) * self.last_bbox[0])
        y1 = int(alpha * new_bbox[1] + (1 - alpha) * self.last_bbox[1])
        x2 = int(alpha * new_bbox[2] + (1 - alpha) * self.last_bbox[2])
        y2 = int(alpha * new_bbox[3] + (1 - alpha) * self.last_bbox[3])
        
        smoothed_bbox = (x1, y1, x2, y2)
        self.last_bbox = smoothed_bbox
        
        return smoothed_bbox
    
    def run(self, max_frames: Optional[int] = None, display: bool = False):
        """
        Run face tracking loop.
        
        Args:
            max_frames: Maximum number of frames to process (None for infinite)
            display: Whether to display results (requires PIL)
        """
        frame_count = 0
        fps_start_time = time.time()
        
        print("\nStarting face tracking...")
        print("Press Ctrl+C to stop")
        
        # Prepare optional pygame display
        use_pygame_display = False
        screen = None
        clock = None
        if display:
            try:
                import pygame
                pygame.init()

                # Determine size from camera if available
                if hasattr(self.camera, 'size'):
                    win_w, win_h = self.camera.size
                else:
                    win_w, win_h = 640, 480

                screen = pygame.display.set_mode((win_w, win_h))
                pygame.display.set_caption('Face Tracker')
                clock = pygame.time.Clock()
                use_pygame_display = True
            except Exception as e:
                print(f"Pygame display not available: {e}. Falling back to PIL image.show()")

        try:
            running = True
            while running:
                # Capture frame
                frame = self.camera.read()
                
                if frame is None:
                    print("Failed to capture frame")
                    continue
                
                # Process frame
                start_time = time.time()
                skin_mask, bbox = self.process_frame(frame)
                process_time = time.time() - start_time
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed > 1.0:
                    fps = frame_count / elapsed
                    print(f"FPS: {fps:.2f} | Processing time: {process_time*1000:.2f}ms | BBox: {bbox}")
                    frame_count = 0
                    fps_start_time = time.time()
                
                # Display (optional) - prefer pygame for realtime display
                if display:
                    if use_pygame_display and screen is not None:
                        try:
                            import pygame
                            # Convert frame (H,W,3) to pygame surface
                            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                            screen.blit(surf, (0, 0))

                            # Draw bounding box
                            if bbox != (0, 0, 0, 0):
                                x1, y1, x2, y2 = bbox
                                rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
                                pygame.draw.rect(screen, (0, 255, 0), rect, 3)

                            # Update display and handle events
                            pygame.display.flip()

                            # Handle events (allow closing window)
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    running = False
                                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                                    running = False

                            # Cap the frame rate slightly
                            if clock is not None:
                                clock.tick(60)
                        except Exception as e:
                            print(f"Realtime pygame display error: {e}")
                            # Fallback to PIL display for a single frame
                            self._display_results(frame, skin_mask, bbox)
                    else:
                        # Use PIL fallback (will open external viewer per frame)
                        self._display_results(frame, skin_mask, bbox)
                
                # Check if max frames reached
                if max_frames is not None and frame_count >= max_frames:
                    break
        
        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            # Cleanup pygame if used
            if use_pygame_display:
                try:
                    import pygame
                    pygame.quit()
                except Exception:
                    pass

            self.camera.release()
            print("Camera released")
    
    def _display_results(self, frame: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        Display tracking results (requires PIL).
        
        Args:
            frame: Original frame
            mask: Skin mask
            bbox: Bounding box
        """
        try:
            from PIL import Image, ImageDraw
            
            # Convert to PIL
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            
            # Draw bounding box
            if bbox != (0, 0, 0, 0):
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
            
            # Show image
            img.show()
        
        except ImportError:
            print("PIL not available for display")
    
    def get_current_frame_and_bbox(self) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Get current frame and face bounding box.
        
        Returns:
            Tuple of (frame, bbox)
        """
        frame = self.camera.read()
        if frame is None:
            return None, (0, 0, 0, 0)
        
        _, bbox = self.process_frame(frame)
        return frame, bbox


if __name__ == "__main__":
    print("=" * 50)
    print("Real-Time Face Tracking")
    print("=" * 50)
    
    # Check if model exists
    model_path = "models/skin_detector_knn.pkl"
    
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        sys.exit(1)
    
    # Ask user about camera
    print("\nCamera options:")
    print("1. Use real webcam (requires pygame)")
    print("2. Use dummy camera (for testing)")
    
    choice = input("Select option (1-2): ").strip()
    use_dummy = (choice == "2")
    
    try:
        # Create tracker
        tracker = FaceTracker(model_path, use_dummy_camera=use_dummy)

        # Run tracking with realtime display (pygame)
        tracker.run(max_frames=None, display=True)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()