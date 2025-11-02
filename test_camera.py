"""
Test camera detection and DroidCam compatibility
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_pygame_cameras():
    """Test pygame camera detection."""
    print("=" * 60)
    print("Testing Pygame Camera Detection")
    print("=" * 60)
    
    try:
        import pygame.camera
        pygame.camera.init()
        
        cameras = pygame.camera.list_cameras()
        
        if not cameras:
            print("\n❌ No cameras detected by pygame!")
            print("\nPossible solutions:")
            print("1. Make sure DroidCam is running and connected")
            print("2. DroidCam may not be compatible with pygame.camera")
            print("3. Try using OpenCV instead (cv2.VideoCapture)")
            return False
        
        print(f"\n✅ Found {len(cameras)} camera(s):")
        for i, cam in enumerate(cameras):
            print(f"  [{i}] {cam}")
        
        # Try to open each camera
        print("\n" + "=" * 60)
        print("Testing Camera Access")
        print("=" * 60)
        
        for i, cam in enumerate(cameras):
            print(f"\nTesting camera [{i}]: {cam}")
            try:
                camera = pygame.camera.Camera(cam, (640, 480))
                camera.start()
                
                # Try to capture a frame
                import time
                time.sleep(1)  # Wait for camera to warm up
                
                image = camera.get_image()
                camera.stop()
                
                print(f"  ✅ Camera [{i}] works! Size: {image.get_size()}")
                
            except Exception as e:
                print(f"  ❌ Camera [{i}] failed: {e}")
        
        return True
        
    except ImportError:
        print("\n❌ Pygame not installed!")
        print("Install with: pip install pygame")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_opencv_fallback():
    """Test OpenCV as fallback for DroidCam."""
    print("\n" + "=" * 60)
    print("Testing OpenCV (Fallback for DroidCam)")
    print("=" * 60)
    
    try:
        import cv2
        
        print("\n✅ OpenCV is installed")
        print("\nTesting camera indices 0-5...")
        
        working_cameras = []
        
        for i in range(6):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"  ✅ Camera {i} works! Resolution: {frame.shape[1]}x{frame.shape[0]}")
                        working_cameras.append(i)
                    else:
                        print(f"  ⚠️  Camera {i} opened but can't read frames")
                    cap.release()
                else:
                    print(f"  ❌ Camera {i} not available")
            except Exception as e:
                print(f"  ❌ Camera {i} error: {e}")
        
        if working_cameras:
            print(f"\n✅ Working cameras found at indices: {working_cameras}")
            print("\nTo use OpenCV instead of pygame, update webcam_capture.py")
            return True
        else:
            print("\n❌ No working cameras found with OpenCV")
            return False
            
    except ImportError:
        print("\n❌ OpenCV not installed")
        print("Install with: pip install opencv-python")
        print("\nNote: OpenCV usually works better with DroidCam!")
        return False


def test_droidcam_detection():
    """Specific tests for DroidCam."""
    print("\n" + "=" * 60)
    print("DroidCam Detection Tips")
    print("=" * 60)
    
    print("""
DroidCam Setup Checklist:
1. ✓ DroidCam app running on your phone
2. ✓ DroidCam Client running on your PC
3. ✓ Connection established (WiFi or USB)
4. ✓ "Video" option enabled in DroidCam Client
5. ✓ No other apps using the camera

DroidCam Camera Locations:
- Windows: Usually appears as "DroidCam Source 3" or similar
- Pygame: May not detect DroidCam virtual camera
- OpenCV: Usually works well with DroidCam (try cv2.VideoCapture(1) or (2))

Recommended Solution for DroidCam:
Use OpenCV instead of pygame for better compatibility!
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CAMERA DETECTION TOOL")
    print("=" * 60)
    
    # Test 1: Pygame
    pygame_ok = test_pygame_cameras()
    
    # Test 2: OpenCV
    opencv_ok = test_opencv_fallback()
    
    # Test 3: DroidCam tips
    test_droidcam_detection()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if pygame_ok:
        print("✅ Pygame camera detection works")
    else:
        print("❌ Pygame camera detection failed")
    
    if opencv_ok:
        print("✅ OpenCV camera detection works - RECOMMENDED FOR DROIDCAM")
    else:
        print("❌ OpenCV not available")
    
    if not pygame_ok and not opencv_ok:
        print("\n⚠️  No working camera method found!")
        print("Solutions:")
        print("1. Install OpenCV: pip install opencv-python")
        print("2. Make sure DroidCam is properly connected")
        print("3. Use dummy camera for testing: python face_tracker.py (select option 2)")
