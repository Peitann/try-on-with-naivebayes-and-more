# Try-On Filter - Visual Workflow

## System Overview
```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRY-ON FILTER APPLICATION                         │
│                                                                       │
│  ┌─────────────────────┐              ┌─────────────────────┐      │
│  │  PYTHON BACKEND     │              │   GODOT FRONTEND    │      │
│  │  (ML Tracking)      │◄────UDP─────►│   (Visualization)   │      │
│  │                     │   Port 9999   │                     │      │
│  │  Face Detection     │              │   AR Overlay        │      │
│  │  BBox Extraction    │              │   Filter Display    │      │
│  └─────────────────────┘              └─────────────────────┘      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Training Workflow
```
╔═══════════════════════════════════════════════════════════════════╗
║                        TRAINING PHASE                              ║
╚═══════════════════════════════════════════════════════════════════╝

Step 1: Data Collection
  ┌──────────┐     ┌──────────────┐     ┌──────────────────┐
  │ Webcam   │────►│ Capture 20+  │────►│ Save to          │
  │ (pygame) │     │ face images  │     │ datasets/        │
  └──────────┘     └──────────────┘     └──────────────────┘

Step 2: Data Labeling
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ Training     │────►│ Mark skin vs │────►│ Save labels  │
  │ images       │     │ non-skin     │     │ (JSON)       │
  └──────────────┘     └──────────────┘     └──────────────┘

Step 3: Feature Extraction
  ┌──────────┐     ┌──────────────┐     ┌──────────────┐
  │ RGB      │────►│ Manual RGB→  │────►│ HSV features │
  │ images   │     │ HSV convert  │     │ (H, S, V)    │
  └──────────┘     └──────────────┘     └──────────────┘

Step 4: Model Training
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ HSV features │────►│ Train KNN/NB │────►│ Save model   │
  │ + labels     │     │ /Decision Tr │     │ (.pkl file)  │
  └──────────────┘     └──────────────┘     └──────────────┘
```

## Runtime Workflow
```
╔═══════════════════════════════════════════════════════════════════╗
║                        RUNTIME PHASE                               ║
╚═══════════════════════════════════════════════════════════════════╝

PYTHON SIDE:                         GODOT SIDE:
─────────────                        ────────────

1. Capture Frame                     8. Receive UDP Data
   ┌──────────┐                         ┌──────────┐
   │ Webcam   │                         │ UDP      │
   │ 640×480  │                         │ Socket   │
   └────┬─────┘                         └────┬─────┘
        │                                    │
        ▼                                    ▼
2. RGB → HSV                          9. Parse BBox
   ┌──────────┐                         ┌──────────┐
   │ Manual   │                         │ JSON     │
   │ Convert  │                         │ Parse    │
   └────┬─────┘                         └────┬─────┘
        │                                    │
        ▼                                    ▼
3. Classify Pixels                    10. Map Coordinates
   ┌──────────┐                          ┌──────────┐
   │ ML Model │                          │ Camera→  │
   │ Predict  │                          │ Screen   │
   └────┬─────┘                          └────┬─────┘
        │                                     │
        ▼                                     ▼
4. Binary Mask                        11. Update Filter
   ┌──────────┐                          ┌──────────┐
   │ Skin=255 │                          │ Position │
   │ Other=0  │                          │ & Scale  │
   └────┬─────┘                          └────┬─────┘
        │                                     │
        ▼                                     ▼
5. Connected Components               12. Render Frame
   ┌──────────┐                          ┌──────────┐
   │ Flood    │                          │ Webcam + │
   │ Fill     │                          │ Overlay  │
   └────┬─────┘                          └──────────┘
        │
        ▼
6. Extract BBox
   ┌──────────┐
   │ (x1, y1, │
   │  x2, y2) │
   └────┬─────┘
        │
        ▼
7. Send UDP
   ┌──────────┐──────UDP Packet──────►
   │ JSON msg │    {bbox, frame_size}
   └──────────┘
```

## Face Tracking Pipeline (Detailed)
```
Frame (640×480×3 RGB)
         │
         ▼
    ┌────────────────┐
    │ Manual RGB→HSV │ ──► H: [0, 360]
    │   Conversion   │     S: [0, 100]
    └────────┬───────┘     V: [0, 100]
             │
             ▼
    ┌─────────────────────┐
    │ Reshape to (307200,3)│
    │  (640×480 pixels)    │
    └────────┬─────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  ML Classification  │
    │  (per-pixel)        │
    │  Input: HSV (3D)    │
    │  Output: 0 or 1     │
    └────────┬─────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Reshape to (480,640)│
    │  Binary Mask        │
    └────────┬─────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  Flood-Fill Algo    │
    │  Label Components   │
    │  Find largest       │
    └────────┬─────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  Extract BBox       │
    │  (x1, y1, x2, y2)   │
    └────────┬─────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  Smooth BBox        │
    │  (EMA filter)       │
    └────────┬─────────────┘
             │
             ▼
        Send to Godot
```

## Connected Component Analysis
```
Binary Mask (480×640):
┌────────────────────┐
│ 0 0 0 0 0 0 0 0   │  0 = Background
│ 0 0 255 255 0 0   │  255 = Skin
│ 0 255 255 255 0   │
│ 0 255 255 255 0   │  ┌──────────┐
│ 0 0 255 255 0 0   │  │Component │
│ 0 0 0 0 0 0 0 0   │  │  Label   │
└────────────────────┘  └────┬─────┘
                              │
         Flood Fill ──────────┤
                              │
                              ▼
                       Find Largest
                              │
                              ▼
                    ┌─────────────────┐
                    │ Bounding Box:   │
                    │ x1 = 100        │
                    │ y1 = 150        │
                    │ x2 = 300        │
                    │ y2 = 400        │
                    └─────────────────┘
```

## Godot Scene Hierarchy
```
MainScene (Node2D)
├── WebcamDisplay (Sprite2D)
│   └── texture: Live camera feed
│
├── FilterOverlay (Sprite2D)
│   ├── position: Follows face bbox
│   ├── scale: Matches face size
│   └── children: 
│       ├── Sticker1 (Sprite2D)
│       ├── Sticker2 (Sprite2D)
│       └── Doodle1 (Line2D)
│
├── CanvasEditor (Control)
│   ├── DrawingCanvas (Panel)
│   │   └── User draws here
│   └── UI (VBoxContainer)
│       ├── ColorPicker (ColorPickerButton)
│       ├── ClearButton (Button)
│       └── SaveButton (Button)
│
└── UI (CanvasLayer)
    └── StatusLabel (Label)
        └── Shows tracking status
```

## Data Flow Diagram
```
┌──────────────────────────────────────────────────────────────┐
│                      DATA FLOW                                │
└──────────────────────────────────────────────────────────────┘

USER INTERACTION                 PROCESSING                OUTPUT
─────────────────               ──────────                ──────

┌──────────┐                    ┌──────────┐           ┌──────────┐
│          │                    │ Python   │           │          │
│  Webcam  │───────Frame───────►│ Tracking │──BBox────►│  Godot   │
│          │     640×480        │ System   │  UDP     │  Display │
└──────────┘                    └──────────┘           └────┬─────┘
                                      │                     │
┌──────────┐                          │                     │
│ Canvas   │                          │                     │
│ Editor   │───────Draw────────────────┼─────────────────────┤
│          │    Stickers/Doodles       │                     │
└──────────┘                           │                     ▼
                                       │              ┌──────────┐
┌──────────┐                           │              │          │
│ Training │                           │              │  Screen  │
│  Data    │───────Labeled────────────►│              │  Output  │
│          │      Images               │              │          │
└──────────┘                           │              └──────────┘
                                       │
                                       ▼
                                ┌──────────┐
                                │  Trained │
                                │  Model   │
                                │  .pkl    │
                                └──────────┘
```

## Performance Metrics
```
┌─────────────────────────────────────────────────────────────┐
│                    PERFORMANCE TARGETS                       │
├─────────────────────────────────────────────────────────────┤
│ Metric              │ Target        │ Typical             │
├─────────────────────┼───────────────┼─────────────────────┤
│ Training Time       │ < 60s         │ 10-30s              │
│ Model Accuracy      │ > 85%         │ 88-96%              │
│ Inference Time      │ < 50ms        │ 20-50ms             │
│ FPS                 │ > 20          │ 20-50               │
│ UDP Latency         │ < 5ms         │ 1-2ms               │
│ End-to-End Latency  │ < 100ms       │ 30-80ms             │
│ Memory Usage        │ < 500MB       │ 200-400MB           │
└─────────────────────────────────────────────────────────────┘
```

## File Sizes (Approximate)
```
├── Python modules       : ~50 KB
├── Godot project        : ~10 KB
├── Training images      : 5-20 MB (20-100 images)
├── Labels (JSON)        : 10-50 KB
├── Trained models       : 1-10 MB (depends on model)
├── Documentation        : ~100 KB
└── Total (without data) : ~200 KB
```

---

**Legend**:
- `────►` : Data flow
- `◄────►` : Bidirectional communication
- `┌─┐`   : Component/Module
- `│`     : Connection
- `▼`     : Flow direction
