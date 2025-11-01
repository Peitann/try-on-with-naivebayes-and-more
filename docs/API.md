# API Documentation

Complete API reference for the Try-On Filter application.

## Python Modules

### image_utils

Manual image processing utilities without OpenCV.

#### `rgb_to_hsv(rgb_image: np.ndarray) -> np.ndarray`

Convert RGB image to HSV color space.

**Parameters**:
- `rgb_image`: numpy array of shape (H, W, 3) with values in [0, 255]

**Returns**:
- HSV image as numpy array with shape (H, W, 3)
  - H: [0, 360]
  - S: [0, 100]
  - V: [0, 100]

**Example**:
```python
import numpy as np
from python_ml_tracking.image_utils import rgb_to_hsv

rgb_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
hsv_img = rgb_to_hsv(rgb_img)
print(hsv_img.shape)  # (480, 640, 3)
```

#### `create_binary_mask(condition_array: np.ndarray) -> np.ndarray`

Create binary mask from boolean condition array.

**Parameters**:
- `condition_array`: Boolean numpy array

**Returns**:
- Binary mask (0 or 255)

#### `morphological_opening(binary_mask: np.ndarray, kernel_size: int = 5) -> np.ndarray`

Manual morphological opening to remove noise.

**Parameters**:
- `binary_mask`: Binary image (0 or 255)
- `kernel_size`: Size of structuring element

**Returns**:
- Cleaned binary mask

---

### connected_components

Manual connected component analysis without OpenCV.

#### `ConnectedComponents` class

Find and analyze connected components in binary images.

**Methods**:

##### `find_components(binary_mask: np.ndarray) -> int`

Find all connected components.

**Parameters**:
- `binary_mask`: Binary image (0 or 255)

**Returns**:
- Number of components found

##### `get_largest_component() -> Tuple[int, Tuple[int, int, int, int]]`

Get largest connected component and its bounding box.

**Returns**:
- Tuple of (component_label, bounding_box)
- bounding_box: (x1, y1, x2, y2)

**Example**:
```python
from python_ml_tracking.connected_components import ConnectedComponents

cc = ConnectedComponents()
num_components = cc.find_components(binary_mask)
label, bbox = cc.get_largest_component()
print(f"Largest component at: {bbox}")
```

#### `find_largest_skin_region(binary_mask: np.ndarray) -> Tuple[int, int, int, int]`

Find largest connected skin region (assumed to be face).

**Parameters**:
- `binary_mask`: Binary skin mask (0 or 255)

**Returns**:
- Bounding box as (x1, y1, x2, y2)

---

### webcam_capture

Webcam capture without OpenCV.

#### `WebcamCapture` class

Cross-platform webcam capture.

**Constructor**:
```python
WebcamCapture(camera_index: int = 0, size: Tuple[int, int] = (640, 480))
```

**Parameters**:
- `camera_index`: Camera device index
- `size`: Frame size (width, height)

**Methods**:

##### `read() -> Optional[np.ndarray]`

Capture a frame from camera.

**Returns**:
- Frame as numpy array (H, W, 3) or None

##### `is_opened() -> bool`

Check if camera is opened.

##### `release()`

Release camera resources.

**Example**:
```python
from python_ml_tracking.webcam_capture import WebcamCapture

cam = WebcamCapture(camera_index=0, size=(640, 480))
if cam.is_opened():
    frame = cam.read()
    if frame is not None:
        print(f"Captured frame: {frame.shape}")
    cam.release()
```

#### `DummyWebcam` class

Dummy webcam for testing.

**Methods**:
- `read()`: Generate dummy frame
- `release()`: No-op
- `is_opened()`: Always returns True

---

### train_model

ML model training for skin detection.

#### `SkinDetectionTrainer` class

Train classical ML models for skin detection.

**Constructor**:
```python
SkinDetectionTrainer(model_type: str = "knn")
```

**Parameters**:
- `model_type`: "knn", "naive_bayes", or "decision_tree"

**Methods**:

##### `extract_features_from_labels(image_dir: str, label_dir: str) -> Tuple[np.ndarray, np.ndarray]`

Extract HSV features from labeled images.

**Parameters**:
- `image_dir`: Directory with training images
- `label_dir`: Directory with label JSON files

**Returns**:
- Tuple of (features, labels)
  - features: (n_samples, 3) HSV values
  - labels: (n_samples,) 1 for skin, 0 for non-skin

##### `train(X: np.ndarray, y: np.ndarray, test_size: float = 0.2)`

Train the classifier.

**Parameters**:
- `X`: Features array (n_samples, 3)
- `y`: Labels array (n_samples,)
- `test_size`: Fraction for testing

##### `evaluate()`

Evaluate trained model on test set.

##### `save_model(output_path: str)`

Save trained model to disk.

##### `load_model(model_path: str)` (static)

Load trained model from disk.

**Example**:
```python
from python_ml_tracking.train_model import SkinDetectionTrainer

trainer = SkinDetectionTrainer(model_type="knn")
X, y = trainer.extract_features_from_labels("datasets/training_images", "datasets/labels")
trainer.train(X, y)
trainer.evaluate()
trainer.save_model("models/my_model.pkl")
```

---

### face_tracker

Real-time face tracking.

#### `FaceTracker` class

Real-time face tracking using trained ML model.

**Constructor**:
```python
FaceTracker(model_path: str, use_dummy_camera: bool = False)
```

**Parameters**:
- `model_path`: Path to trained model pickle file
- `use_dummy_camera`: Use dummy camera for testing

**Methods**:

##### `process_frame(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]`

Process single frame and extract face bounding box.

**Parameters**:
- `frame`: RGB image (H, W, 3)

**Returns**:
- Tuple of (skin_mask, bounding_box)
  - skin_mask: Binary mask
  - bounding_box: (x1, y1, x2, y2)

##### `run(max_frames: Optional[int] = None, display: bool = False)`

Run face tracking loop.

**Parameters**:
- `max_frames`: Max frames to process
- `display`: Show results (requires PIL)

##### `get_current_frame_and_bbox() -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]`

Get current frame and face bbox.

**Returns**:
- Tuple of (frame, bbox)

**Example**:
```python
from python_ml_tracking.face_tracker import FaceTracker

tracker = FaceTracker("models/skin_detector_knn.pkl")
frame, bbox = tracker.get_current_frame_and_bbox()
print(f"Face at: {bbox}")
```

---

### communication

Python-Godot communication.

#### `UDPServer` class

UDP server to send tracking data to Godot.

**Constructor**:
```python
UDPServer(host: str = "127.0.0.1", port: int = 9999)
```

**Methods**:

##### `send_bbox(bbox: Tuple[int, int, int, int], frame_size: Tuple[int, int] = (640, 480))`

Send bounding box data as JSON.

**Parameters**:
- `bbox`: (x1, y1, x2, y2)
- `frame_size`: (width, height)

##### `send_bbox_binary(bbox: Tuple[int, int, int, int])`

Send bounding box as binary data (16 bytes).

##### `close()`

Close socket.

**Example**:
```python
from python_ml_tracking.communication import UDPServer

server = UDPServer(host="127.0.0.1", port=9999)
server.send_bbox((100, 150, 300, 400), frame_size=(640, 480))
server.close()
```

#### `TrackingServer` class

Combined tracking and communication server.

**Constructor**:
```python
TrackingServer(tracker, comm_protocol: str = "udp", host: str = "127.0.0.1", port: int = 9999)
```

**Parameters**:
- `tracker`: FaceTracker instance
- `comm_protocol`: "udp" or "websocket"
- `host`: Server host
- `port`: Server port

**Methods**:

##### `start()`

Start server in background thread.

##### `stop()`

Stop server.

**Example**:
```python
from python_ml_tracking.face_tracker import FaceTracker
from python_ml_tracking.communication import TrackingServer

tracker = FaceTracker("models/skin_detector_knn.pkl")
server = TrackingServer(tracker, comm_protocol="udp", port=9999)
server.start()
# ... run for some time ...
server.stop()
```

---

## Godot Scripts (GDScript)

### MainScene.gd

Main scene controller.

#### Constants

```gdscript
const UDP_PORT = 9999
const UDP_HOST = "127.0.0.1"
```

#### Methods

##### `setup_udp_connection()`

Setup UDP socket to receive tracking data.

##### `receive_tracking_data()`

Receive and parse face tracking data from Python.

##### `update_filter_position()`

Update filter overlay position based on face bbox.

##### `update_status(message: String)`

Update status label text.

---

### FilterOverlay.gd

Filter overlay controller.

#### Methods

##### `load_sticker(sticker_path: String)`

Load and add sticker to filter.

**Parameters**:
- `sticker_path`: Path to sticker image

##### `add_doodle(points: PackedVector2Array, color: Color, width: float)`

Add doodle line to filter.

**Parameters**:
- `points`: Array of 2D points
- `color`: Line color
- `width`: Line width

##### `clear_filter()`

Clear all filter elements.

---

### CanvasEditor.gd

Canvas editor for drawing.

#### Signals

```gdscript
signal filter_updated(filter_data)
```

#### Methods

##### `export_filter_data() -> Dictionary`

Export filter data for application.

**Returns**:
- Dictionary with strokes and metadata

---

## UDP Protocol

### JSON Message Format

```json
{
  "bbox": {
    "x1": 100,
    "y1": 150,
    "x2": 300,
    "y2": 400
  },
  "frame_size": {
    "width": 640,
    "height": 480
  },
  "timestamp": 1234567890.123
}
```

### Binary Message Format

16 bytes total:
- Bytes 0-3: x1 (int32)
- Bytes 4-7: y1 (int32)
- Bytes 8-11: x2 (int32)
- Bytes 12-15: y2 (int32)

---

## Command Line Interface

### main.py

Main application runner.

```bash
python python_ml_tracking/main.py [OPTIONS]
```

**Options**:
- `--model PATH`: Path to trained model (default: models/skin_detector_knn.pkl)
- `--dummy-camera`: Use dummy camera for testing
- `--host HOST`: UDP host (default: 127.0.0.1)
- `--port PORT`: UDP port (default: 9999)
- `--protocol PROTO`: Communication protocol (udp/websocket)

**Example**:
```bash
python python_ml_tracking/main.py --model models/skin_detector_knn.pkl --port 9999
```

---

## Error Codes

- `0`: Success
- `1`: General error
- `2`: Model not found
- `3`: Camera initialization failed
- `4`: UDP socket bind failed

---

**Last Updated**: November 2025
