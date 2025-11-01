"""
Communication module for sending tracking data from Python to Godot
Supports both UDP and WebSocket protocols
"""

import socket
import json
import threading
import time
from typing import Tuple, Optional, Callable
import struct


class UDPServer:
    """
    Simple UDP server to send face tracking data to Godot.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        """
        Initialize UDP server.
        
        Args:
            host: Server host address
            port: Server port
        """
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        
        print(f"UDP Server initialized on {host}:{port}")
    
    def send_bbox(self, bbox: Tuple[int, int, int, int], frame_size: Tuple[int, int] = (640, 480)):
        """
        Send bounding box data to client.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_size: Frame dimensions (width, height)
        """
        try:
            # Prepare data packet
            data = {
                "bbox": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3]
                },
                "frame_size": {
                    "width": frame_size[0],
                    "height": frame_size[1]
                },
                "timestamp": time.time()
            }
            
            # Convert to JSON and send
            message = json.dumps(data).encode('utf-8')
            self.sock.sendto(message, (self.host, self.port))
        
        except Exception as e:
            print(f"Error sending UDP packet: {e}")
    
    def send_bbox_binary(self, bbox: Tuple[int, int, int, int]):
        """
        Send bounding box as binary data (more efficient).
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
        
        Format: 4 integers (16 bytes total)
        """
        try:
            # Pack as 4 integers
            message = struct.pack('iiii', *bbox)
            self.sock.sendto(message, (self.host, self.port))
        except Exception as e:
            print(f"Error sending binary UDP packet: {e}")
    
    def close(self):
        """Close the socket."""
        self.sock.close()
        print("UDP Server closed")


class TrackingServer:
    """
    Server that combines face tracking and communication.
    Continuously sends tracking data to Godot.
    """
    
    def __init__(self, tracker, comm_protocol: str = "udp",
                 host: str = "127.0.0.1", port: int = 9999):
        """
        Initialize tracking server.
        
        Args:
            tracker: FaceTracker instance
            comm_protocol: Communication protocol ("udp" or "websocket")
            host: Server host
            port: Server port
        """
        self.tracker = tracker
        self.protocol = comm_protocol
        self.running = False
        self.thread = None
        
        # Initialize communication
        if comm_protocol == "udp":
            self.comm = UDPServer(host, port)
        else:
            raise NotImplementedError(f"Protocol {comm_protocol} not implemented yet")
    
    def start(self):
        """Start the server in a separate thread."""
        if self.running:
            print("Server already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        print("Tracking server started")
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        
        self.comm.close()
        print("Tracking server stopped")
    
    def _run_loop(self):
        """Main server loop."""
        frame_count = 0
        fps_start = time.time()
        
        while self.running:
            try:
                # Get current frame and bbox from tracker
                frame, bbox = self.tracker.get_current_frame_and_bbox()
                
                if frame is not None:
                    # Send bbox to Godot
                    frame_size = (frame.shape[1], frame.shape[0])
                    self.comm.send_bbox(bbox, frame_size)
                    
                    # Calculate FPS
                    frame_count += 1
                    elapsed = time.time() - fps_start
                    if elapsed > 2.0:
                        fps = frame_count / elapsed
                        print(f"Tracking Server FPS: {fps:.2f} | BBox: {bbox}")
                        frame_count = 0
                        fps_start = time.time()
                
                # Small delay to avoid overwhelming the network
                time.sleep(0.01)  # ~100 FPS max
            
            except Exception as e:
                print(f"Error in tracking loop: {e}")
                time.sleep(0.1)


class BBoxReceiver:
    """
    Receiver class for testing (receives bbox data from Python).
    This can be used in Godot's GDScript equivalent.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        """
        Initialize UDP receiver.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(1.0)  # 1 second timeout
        
        print(f"UDP Receiver listening on {host}:{port}")
    
    def receive_bbox(self, timeout: Optional[float] = None) -> Optional[dict]:
        """
        Receive bounding box data.
        
        Args:
            timeout: Receive timeout in seconds
        
        Returns:
            Dictionary with bbox data or None if timeout
        """
        if timeout is not None:
            self.sock.settimeout(timeout)
        
        try:
            data, addr = self.sock.recvfrom(1024)
            message = json.loads(data.decode('utf-8'))
            return message
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None
    
    def receive_bbox_binary(self, timeout: Optional[float] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Receive binary bounding box data.
        
        Returns:
            Tuple (x1, y1, x2, y2) or None
        """
        if timeout is not None:
            self.sock.settimeout(timeout)
        
        try:
            data, addr = self.sock.recvfrom(16)
            bbox = struct.unpack('iiii', data)
            return bbox
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error receiving binary data: {e}")
            return None
    
    def close(self):
        """Close the socket."""
        self.sock.close()
        print("Receiver closed")


if __name__ == "__main__":
    print("=" * 50)
    print("Communication Module Test")
    print("=" * 50)
    
    print("\nSelect test mode:")
    print("1. Test sender (sends dummy bbox data)")
    print("2. Test receiver (receives bbox data)")
    
    choice = input("Select mode (1-2): ").strip()
    
    if choice == "1":
        # Test sender
        print("\nStarting UDP sender...")
        sender = UDPServer(host="127.0.0.1", port=9999)
        
        print("Sending test bounding boxes...")
        print("Run receiver in another terminal to see the data")
        
        try:
            for i in range(100):
                # Send dummy bbox
                bbox = (100 + i*2, 150 + i, 300 + i*2, 400 + i)
                sender.send_bbox(bbox, frame_size=(640, 480))
                print(f"Sent: {bbox}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            sender.close()
    
    elif choice == "2":
        # Test receiver
        print("\nStarting UDP receiver...")
        receiver = BBoxReceiver(host="127.0.0.1", port=9999)
        
        print("Listening for bounding box data...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                data = receiver.receive_bbox(timeout=1.0)
                if data is not None:
                    bbox = data['bbox']
                    print(f"Received: ({bbox['x1']}, {bbox['y1']}, {bbox['x2']}, {bbox['y2']})")
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            receiver.close()
