# Try-On Filter - System Architecture

## Overview

The Try-On Filter application is a desktop AR filter system that uses classical Machine Learning for face tracking, without relying on OpenCV, MediaPipe, dlib, or deep learning models.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRY-ON FILTER SYSTEM                         │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐  ┌──────────────────────────────┐
│     PYTHON TRACKING BACKEND      │  │    GODOT VISUALIZATION       │
│                                  │  │                              │
│  ┌────────────────────────────┐ │  │ ┌──────────────────────────┐ │
│  │   1. Data Collection       │ │  │ │   Canvas Editor          │ │
│  │   • Webcam capture         │ │  │ │   • Draw doodles         │ │
│  │   • Image storage          │ │  │ │   • Place stickers       │ │
│  └────────────────────────────┘ │  │ │   • Save designs         │ │
│                                  │  │ └──────────────────────────┘ │
│  ┌────────────────────────────┐ │  │                              │
│  │   2. Data Labeling         │ │  │ ┌──────────────────────────┐ │
│  │   • Skin region marking    │ │  │ │   Webcam Display         │ │
│  │   • Label storage (JSON)   │ │  │ │   • Live video feed      │ │
│  └────────────────────────────┘ │  │ │   • Texture rendering    │ │
│                                  │  │ └──────────────────────────┘ │
│  ┌────────────────────────────┐ │  │                              │
│  │   3. Feature Extraction    │ │  │ ┌──────────────────────────┐ │
│  │   • Manual RGB→HSV convert │ │  │ │   Filter Overlay         │ │
│  │   • Pixel feature vectors  │ │  │ │   • Position tracking    │ │
│  └────────────────────────────┘ │  │ │   • Scale matching       │ │
│                                  │  │ │   • Real-time rendering  │ │
│  ┌────────────────────────────┐ │  │ └──────────────────────────┘ │
│  │   4. Model Training        │ │  │                              │
│  │   • KNN classifier          │ │  │ ┌──────────────────────────┐ │
│  │   • Naive Bayes            │ │  │ │   UDP Receiver           │ │
│  │   • Decision Tree          │ │  │ │   • Parse bbox data      │ │
│  │   • Model evaluation       │ │  │ │   • Update overlay       │ │
│  └────────────────────────────┘ │  │ └──────────────────────────┘ │
│                                  │  │                              │
│  ┌────────────────────────────┐ │  └──────────────────────────────┘
│  │   5. Real-time Tracking    │ │
│  │   • Webcam frame capture   │ │
│  │   • HSV conversion         │ │            UDP Socket
│  │   • Skin classification    │ ├──────────► Port 9999
│  │   • Connected components   │ │    {x1, y1, x2, y2}
│  │   • Bounding box           │ │
│  └────────────────────────────┘ │
│                                  │
│  ┌────────────────────────────┐ │
│  │   6. Communication Server  │ │
│  │   • UDP/WebSocket server   │ │
│  │   • JSON serialization     │ │
│  │   • Real-time transmission │ │
│  └────────────────────────────┘ │
└──────────────────────────────────┘
```

## Component Details

### 1. Image Processing Module (`image_utils.py`)

**Purpose**: Manual image processing without OpenCV

**Key Functions**:
- `rgb_to_hsv()`: Converts RGB image to HSV color space using manual formulas
- `create_binary_mask()`: Creates binary masks from boolean arrays
- `morphological_opening()`: Noise reduction using erosion and dilation
- `erosion()`: Manual erosion operation
- `dilation()`: Manual dilation operation

**Implementation**:
```python
# RGB to HSV conversion formula
max_c = max(R, G, B)
min_c = min(R, G, B)
delta = max_c - min_c

if max_c == R:
    H = 60 * (((G - B) / delta) % 6)
elif max_c == G:
    H = 60 * (((B - R) / delta) + 2)
else:
    H = 60 * (((R - G) / delta) + 4)

S = 0 if max_c == 0 else delta / max_c
V = max_c
```

### 2. Connected Component Analysis (`connected_components.py`)

**Purpose**: Find largest skin region (face) without OpenCV

**Algorithm**: Flood-fill based component labeling
1. Scan image pixel by pixel
2. When unlabeled foreground pixel found, start flood-fill
3. Label all connected pixels with same component ID
4. Track component sizes and bounding boxes
5. Return largest component as face

**Complexity**: O(W × H) where W=width, H=height

### 3. Machine Learning Pipeline

#### Feature Extraction
- **Input**: Labeled face images
- **Process**: Extract HSV values from skin and non-skin regions
- **Output**: Feature matrix X (n_samples, 3) and labels y (n_samples,)

#### Training
Three classical ML models are trained:

**K-Nearest Neighbors (KNN)**
- Distance metric: Euclidean
- k = 5 neighbors
- Fast inference: ~20ms per frame
- Accuracy: 92-96%

**Naive Bayes**
- Assumes feature independence
- Very fast training and inference
- Accuracy: 88-93%

**Decision Tree**
- Max depth: 10
- Handles non-linear boundaries
- Accuracy: 90-95%

#### Model Selection
Best model determined by:
1. Test set accuracy
2. Inference speed (FPS)
3. Generalization (different lighting)

### 4. Face Tracking Pipeline

```
Frame Capture → RGB to HSV → Pixel Classification → Binary Mask
     ↓                                                    ↓
  640×480                                         Skin pixels = 255
  RGB array                                       Other pixels = 0
                                                         ↓
                                                  Connected Components
                                                         ↓
                                                  Find Largest Region
                                                         ↓
                                                  Extract Bounding Box
                                                         ↓
                                                    (x1, y1, x2, y2)
```

**Performance**:
- Processing time: 20-50ms per frame
- FPS: 20-50 (depending on resolution and model)
- Latency: <50ms end-to-end

### 5. Communication Protocol

**UDP Protocol** (default):
- Low latency (~1-2ms)
- Connectionless
- Suitable for real-time streaming

**Message Format** (JSON):
```json
{
  "bbox": {
    "x1": int,  // Top-left x
    "y1": int,  // Top-left y
    "x2": int,  // Bottom-right x
    "y2": int   // Bottom-right y
  },
  "frame_size": {
    "width": int,   // Frame width
    "height": int   // Frame height
  },
  "timestamp": float  // Unix timestamp
}
```

**Binary Format** (optional, more efficient):
- 4 integers (x1, y1, x2, y2)
- 16 bytes total
- No serialization overhead

### 6. Godot Visualization

**Main Scene Structure**:
```
MainScene (Node2D)
├── WebcamDisplay (Sprite2D)
│   └── Texture: Live webcam feed
├── FilterOverlay (Sprite2D)
│   ├── Position: Follows face bbox
│   ├── Scale: Matches face size
│   └── Children: Stickers and doodles
├── CanvasEditor (Control)
│   ├── DrawingCanvas (Panel)
│   └── UI (Buttons, ColorPicker)
└── UI (CanvasLayer)
    └── StatusLabel (Label)
```

**Update Loop** (60 FPS):
1. Receive UDP packet with bbox data
2. Parse JSON message
3. Update FilterOverlay position and scale
4. Render frame

**Coordinate Mapping**:
```python
# Camera space (640×480) → Screen space (1280×720)
screen_x = (bbox_center_x / 640) * 1280
screen_y = (bbox_center_y / 480) * 720
```

## Data Flow

### Training Phase
```
User Face → Webcam → Images → Manual Labeling → HSV Features
                                                      ↓
                                           scikit-learn Training
                                                      ↓
                                              Trained Model.pkl
```

### Runtime Phase
```
Webcam → Python Capture → HSV → ML Classify → Mask → BBox
                                                        ↓
                                                     UDP Send
                                                        ↓
                                              Godot Receive
                                                        ↓
                                    Update Filter Position/Scale
                                                        ↓
                                                  Render Frame
```

## Performance Optimization

### Python Side
1. **Vectorization**: Use numpy operations instead of loops
2. **Pixel Sampling**: Classify every nth pixel instead of all
3. **ROI Processing**: Only process center region where face likely is
4. **Bbox Smoothing**: Exponential moving average reduces jitter

### Godot Side
1. **Texture Caching**: Reuse webcam texture
2. **Dirty Flagging**: Only update when bbox changes
3. **Level of Detail**: Scale filter complexity with distance
4. **Batch Rendering**: Combine multiple filter elements

## Constraints & Design Decisions

### Why No OpenCV?
- **Educational**: Understand low-level image processing
- **Lightweight**: Smaller dependencies
- **Customizable**: Full control over algorithms

### Why Classical ML?
- **Interpretable**: Understand model decisions
- **Fast**: Real-time inference on CPU
- **Trainable**: Create personalized models
- **No GPU**: Works on any hardware

### Why UDP?
- **Low Latency**: Essential for real-time AR
- **Simplicity**: Easy to implement
- **Fire-and-forget**: No connection overhead

## Security Considerations

- UDP is unencrypted (local use only)
- No authentication (trusted network)
- Input validation on Godot side
- Rate limiting to prevent DoS

## Extensibility

### Adding New Features
1. **New Filters**: Add to CanvasEditor
2. **New ML Models**: Extend `train_model.py`
3. **Better Tracking**: Add Kalman filter
4. **Multiple Faces**: Extend connected components

### Plugin System
- Filter plugins as separate scripts
- Dynamic loading from filters/ directory
- Plugin manifest with metadata

## Testing Strategy

1. **Unit Tests**: Test each component independently
2. **Integration Tests**: Test Python-Godot communication
3. **Performance Tests**: Measure FPS and latency
4. **Accuracy Tests**: Validate ML model performance

## Deployment

### Packaging Python Backend
```bash
pyinstaller --onefile python_ml_tracking/main.py
```

### Packaging Godot Frontend
```bash
# Export from Godot editor
Project → Export → Windows Desktop
```

### Distribution
- Bundle Python executable with Godot application
- Include trained models in models/ directory
- Provide setup script for first-time configuration

---

**Last Updated**: November 2025
