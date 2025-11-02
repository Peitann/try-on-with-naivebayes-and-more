# Godot Setup Guide

## 🎮 Setting Up Godot Frontend

### Prerequisites
- Godot Engine 4.2+
- Python backend already running (see main README.md)

### Quick Start

1. **Open Godot Project**
   ```
   - Launch Godot Engine
   - Click "Import"
   - Navigate to: godot_project/project.godot
   - Click "Import & Edit"
   ```

2. **Start Python Backend First**
   ```bash
   # Terminal 1: Start Python face tracking server
   python python_ml_tracking/main.py --camera-index 3
   ```
   
   Wait until you see:
   ```
   ============================================================
   Tracking server is running!
   Sending face bounding box data to Godot...
   ============================================================
   ```

3. **Run Godot Scene**
   ```
   - In Godot Editor, press F5 (or click Play button)
   - Scene will open and start receiving data from Python
   ```

### What You Should See

**In Godot Window:**
- Left side: Canvas editor for drawing filters
- Center: Webcam area (gradient placeholder)
- Green semi-transparent box: Face bounding box position
- Status text (top-left): Current tracking status

**Expected Status Messages:**
- Initial: `"Waiting for tracking data..."`
- When tracking: `"Tracking active | BBox: (x1, y1, x2, y2)"`

### Understanding the Display

Since Godot doesn't have native webcam support, we show:

1. **Gradient Placeholder** - Represents camera viewport (640x480)
2. **Green Rectangle** - Shows detected face bounding box position
3. **Filter Overlay** - AR filters will appear here (centered on face)

### Camera Not Showing?

**This is expected!** Godot cannot directly access your webcam. The gradient you see is a placeholder that represents the camera viewport.

**The important part is:**
- ✅ Python backend detects face via DroidCam
- ✅ Python sends bounding box coordinates to Godot via UDP
- ✅ Godot displays green box showing face position
- ✅ Filters overlay on the green box position

### Troubleshooting

**Problem: Status stays "Waiting for tracking data..."**

Solution:
```bash
# Check Python backend is running:
# Terminal should show:
Tracking Server FPS: 1.24 | BBox: (96, 362, 639, 478)

# If not, restart Python backend
python python_ml_tracking/main.py --camera-index 3
```

**Problem: Green box not moving**

Solution:
- Make sure your face is visible in DroidCam
- Check Python terminal shows changing BBox coordinates
- Godot receives data on port 9999 (UDP)

**Problem: Godot won't start**

Solution:
```
1. Check Godot version (must be 4.2+)
2. Re-import project: Project > Reload Current Project
3. Check console for errors (View > Output)
```

### Adding Filters

1. **Use Canvas Editor (left panel)**
   - Choose color from color picker
   - Draw on the canvas area
   - Your drawing will appear as overlay on detected face

2. **Button Functions:**
   - "Clear Canvas" - Removes all drawings
   - "Save Filter" - Saves current filter design (TODO)

### Architecture

```
Python Backend          UDP (port 9999)         Godot Frontend
┌──────────────┐                              ┌───────────────┐
│  DroidCam    │                              │ MainScene.gd  │
│   Camera     │                              │               │
└──────┬───────┘                              │ - Receives    │
       │                                      │   UDP data    │
       ▼                                      │               │
┌──────────────┐      BBox Coordinates       │ - Updates     │
│ Face Tracker │ ────────────────────────────▶│   green box   │
│  (ML Model)  │      {"x1":250, "y1":322,   │               │
└──────────────┘       "x2":541, "y2":455}   │ - Positions   │
                                              │   filters     │
                                              └───────────────┘
```

### Performance

- **FPS**: ~1-1.5 FPS (limited by ML inference speed)
- **Latency**: ~50-100ms network delay
- **Smooth**: Bounding box has smoothing applied

### Next Steps

1. ✅ Verify green box follows your face movement
2. ✅ Test drawing on canvas and see it overlay
3. 🔄 Implement actual webcam feed (requires camera plugin)
4. 🔄 Add more filter options (stickers, effects)
5. 🔄 Save/load filter designs

### Optional: Real Webcam Feed in Godot

To show actual webcam feed instead of gradient:

**Option 1: Use Godot Camera Plugin**
```
1. Install plugin: https://github.com/you-win/godot-webcam
2. Update WebcamDisplay to use plugin texture
```

**Option 2: Stream from Python**
```python
# Modify Python backend to send JPEG frames via UDP
# Update Godot to decode and display frames
# (Not recommended - high bandwidth usage)
```

**Option 3: Use SharedMemory**
```python
# Python writes frames to shared memory
# Godot reads frames from shared memory
# (Best performance, platform-dependent)
```

For this project, **Option 1 (Camera Plugin)** is recommended for production use.

### Debug Mode

Enable debug output in Godot:
```gdscript
# In MainScene.gd, add at top:
const DEBUG = true

# Then add debug prints:
if DEBUG:
    print("Received bbox: ", last_bbox)
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No green box | Python not running | Start Python backend first |
| Box not moving | Face not detected | Check DroidCam, improve lighting |
| Filters not showing | Canvas empty | Draw something on canvas first |
| Port conflict | Port 9999 in use | Change port in both Python and Godot |

---

## 🎯 Success Checklist

- [ ] Python backend running with DroidCam
- [ ] Godot scene opens without errors
- [ ] Status shows "Tracking active"
- [ ] Green box visible on gradient
- [ ] Green box follows face movement
- [ ] Can draw on canvas editor
- [ ] Drawings appear as overlay

If all checked ✅ - Your Try-On Filter system is working! 🎉
