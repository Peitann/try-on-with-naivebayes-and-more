"""
Main application runner
Combines face tracking and communication to send data to Godot
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_ml_tracking.face_tracker import FaceTracker
from python_ml_tracking.communication import TrackingServer


def main():
    """Main entry point for the tracking application."""
    parser = argparse.ArgumentParser(description="Try-On Filter Face Tracking")
    
    # Try to find available trained models
    model_options = [
        "models/skin_detector_decision_tree.pkl",
        "models/skin_detector_naive_bayes.pkl",
        "models/skin_detector_knn.pkl"
    ]
    
    default_model = None
    for path in model_options:
        if os.path.exists(path):
            default_model = path
            break
    
    if default_model is None:
        default_model = "models/skin_detector_knn.pkl"  # Fallback
    
    parser.add_argument("--model", type=str, default=default_model,
                       help="Path to trained model")
    parser.add_argument("--camera-index", type=int, default=0,
                       help="Camera device index (0=default, 3-4=DroidCam)")
    parser.add_argument("--dummy-camera", action="store_true",
                       help="Use dummy camera for testing")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Communication host")
    parser.add_argument("--port", type=int, default=9999,
                       help="Communication port")
    parser.add_argument("--protocol", type=str, default="udp",
                       choices=["udp", "websocket"],
                       help="Communication protocol")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Try-On Filter - Face Tracking System")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Camera: {'Dummy' if args.dummy_camera else f'Real (index {args.camera_index})'}")
    print(f"Communication: {args.protocol.upper()} on {args.host}:{args.port}")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\nError: Model not found at {args.model}")
        print("Please train the model first using train_model.py")
        return 1
    
    try:
        # Initialize face tracker
        print("\nInitializing face tracker...")
        
        if args.dummy_camera:
            tracker = FaceTracker(args.model, use_dummy_camera=True)
        else:
            # Use custom camera index for DroidCam support
            import pickle
            from python_ml_tracking.webcam_capture import WebcamCapture
            
            tracker = FaceTracker.__new__(FaceTracker)
            
            # Load model
            with open(args.model, 'rb') as f:
                tracker.model = pickle.load(f)
            
            # Initialize camera with custom index
            tracker.camera = WebcamCapture(camera_index=args.camera_index, size=(640, 480))
            
            if not tracker.camera.is_opened():
                raise RuntimeError(f"Failed to open camera at index {args.camera_index}!")
            
            tracker.last_bbox = (0, 0, 0, 0)
            tracker.smoothing_factor = 0.3
        
        # Initialize tracking server
        print("Starting tracking server...")
        server = TrackingServer(tracker, comm_protocol=args.protocol,
                               host=args.host, port=args.port)
        server.start()
        
        print("\n" + "=" * 60)
        print("Tracking server is running!")
        print("Sending face bounding box data to Godot...")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        # Keep running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        
        # Cleanup
        server.stop()
        tracker.camera.release()
        
        print("Goodbye!")
        return 0
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
