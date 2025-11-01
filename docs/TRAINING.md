# ML Training Guide

Complete guide for training the skin detection classifier for face tracking.

## Overview

The training process involves:
1. Data collection (capture face images)
2. Data labeling (mark skin vs non-skin regions)
3. Feature extraction (convert to HSV)
4. Model training (KNN, Naive Bayes, Decision Tree)
5. Model evaluation
6. Model selection

## Requirements

- Python 3.8+
- 20-100 face images
- Good variety in lighting conditions
- Clear face visibility

## Step-by-Step Training

### Step 1: Collect Training Images

#### Option A: Using Screen Capture (Testing)
```bash
python python_ml_tracking/data_collector.py
# Select option 1
```

1. Position a photo/video of your face on screen
2. Script captures 20 images at 1-second intervals
3. Images saved to `datasets/training_images/`

#### Option B: Using Pygame Camera (Real Webcam)
```bash
python python_ml_tracking/data_collector.py
# Select option 2
```

1. Ensure webcam is connected
2. pygame will capture 20 images
3. Vary your pose and lighting

#### Option C: Manual Collection
```bash
# Add images manually to datasets/training_images/
# Supported formats: PNG, JPG, JPEG
```

**Best Practices**:
- Capture images in different lighting (bright, dim, natural, artificial)
- Include various angles (front, slight left, slight right)
- Ensure face is clearly visible
- Avoid excessive motion blur
- 640×480 resolution recommended

### Step 2: Label Training Data

#### Automatic Labeling (Quick Start)
```bash
python python_ml_tracking/labeling_tool.py
# Select option 1
```

This creates labels automatically:
- **Skin regions**: Center 40% of image (assumed to be face)
- **Non-skin regions**: Four corners (background)

**Pros**:
- Very fast (seconds)
- Good starting point

**Cons**:
- Less accurate
- May include non-face areas

#### Manual Labeling (Higher Accuracy)
```python
from python_ml_tracking.labeling_tool import LabelingTool

tool = LabelingTool()
tool.load_image("datasets/training_images/train_0000.png")

# Mark skin regions (face, neck, hands)
tool.add_skin_region((100, 150, 300, 400))  # (x1, y1, x2, y2)
tool.add_skin_region((280, 400, 320, 450))  # neck

# Mark non-skin regions (background, clothes, hair)
tool.add_non_skin_region((0, 0, 50, 50))      # top-left background
tool.add_non_skin_region((590, 0, 640, 50))   # top-right background
tool.add_non_skin_region((150, 100, 250, 140)) # hair

# Save labels
tool.save_labels()
```

**Label Format** (JSON):
```json
{
  "image_path": "datasets/training_images/train_0000.png",
  "image_size": [640, 480],
  "skin_regions": [
    [100, 150, 300, 400],
    [280, 400, 320, 450]
  ],
  "non_skin_regions": [
    [0, 0, 50, 50],
    [590, 0, 640, 50],
    [150, 100, 250, 140]
  ]
}
```

**Labeling Tips**:
- Mark diverse skin tones (face, neck, hands)
- Include shadow regions on face
- Mark non-skin: background, clothes, hair, accessories
- Balance skin vs non-skin samples (~50/50 ratio)
- Use larger regions to get more samples

### Step 3: Train Models

#### Train All Models
```bash
python python_ml_tracking/train_model.py
```

This trains three models and saves them to `models/`:
- `skin_detector_knn.pkl`
- `skin_detector_naive_bayes.pkl`
- `skin_detector_decision_tree.pkl`

#### Train Specific Model
```python
from python_ml_tracking.train_model import SkinDetectionTrainer

# Train KNN
trainer = SkinDetectionTrainer(model_type="knn")
X, y = trainer.extract_features_from_labels(
    "datasets/training_images",
    "datasets/labels"
)
trainer.train(X, y, test_size=0.2)
trainer.evaluate()
trainer.save_model("models/skin_detector_knn.pkl")
```

### Step 4: Evaluate Models

Training script automatically prints evaluation metrics:

```
==================================================
Model Evaluation
==================================================
Model type: knn
Training accuracy: 0.9542
Test accuracy: 0.9318

Classification Report (Test Set):
              precision    recall  f1-score   support

    Non-Skin       0.94      0.92      0.93     15234
        Skin       0.93      0.94      0.93     15121

    accuracy                           0.93     30355
   macro avg       0.93      0.93      0.93     30355
weighted avg       0.93      0.93      0.93     30355

Confusion Matrix (Test Set):
[[14023  1211]
 [  860 14261]]
[[TN FP]
 [FN TP]]
```

**Interpretation**:
- **Accuracy**: Overall correctness (target: >90%)
- **Precision**: Of predicted skin, how much is actually skin (target: >90%)
- **Recall**: Of actual skin, how much was detected (target: >90%)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**:
  - TN (True Negative): Correctly classified non-skin
  - FP (False Positive): Non-skin classified as skin
  - FN (False Negative): Skin classified as non-skin
  - TP (True Positive): Correctly classified skin

### Step 5: Select Best Model

Compare models based on:

1. **Test Accuracy**: Higher is better
2. **Inference Speed**: Run face tracker and check FPS
3. **Generalization**: Test in different lighting conditions

**Model Comparison**:

| Model | Accuracy | Speed | Memory | Robustness |
|-------|----------|-------|--------|------------|
| KNN (k=5) | 92-96% | Medium | High | Good |
| Naive Bayes | 88-93% | Fast | Low | Fair |
| Decision Tree | 90-95% | Fast | Medium | Good |

**Recommendation**:
- **Best Overall**: KNN (k=5)
- **Fastest**: Naive Bayes
- **Most Stable**: Decision Tree

### Step 6: Test in Real-time

```bash
python python_ml_tracking/face_tracker.py
```

Monitor:
- FPS (target: >20 FPS)
- Bounding box stability (minimal jitter)
- False detections (should be rare)

## Improving Model Performance

### More Training Data
```bash
# Collect additional images
python python_ml_tracking/data_collector.py

# Label new images
python python_ml_tracking/labeling_tool.py

# Retrain models
python python_ml_tracking/train_model.py
```

### Tune Hyperparameters

#### KNN
```python
from sklearn.neighbors import KNeighborsClassifier

# Try different k values
for k in [3, 5, 7, 9, 11]:
    model = KNeighborsClassifier(n_neighbors=k)
    # ... train and evaluate
```

Larger k → More smooth, less noise, slower
Smaller k → More sensitive, faster, more noise

#### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

# Try different max depths
for depth in [5, 10, 15, 20, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    # ... train and evaluate
```

Deeper tree → More complex, may overfit
Shallow tree → Simpler, may underfit

### Handle Class Imbalance

If you have unequal skin vs non-skin samples:

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)

# Use in models that support it
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(class_weight='balanced')
```

### Feature Engineering

Try additional features:

```python
def extract_extended_features(hsv_image):
    """Extract HSV + spatial features."""
    h, w, _ = hsv_image.shape
    
    # HSV features
    hsv_features = hsv_image.reshape(-1, 3)
    
    # Spatial coordinates (normalized)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_norm = (x_coords / w).reshape(-1, 1)
    y_norm = (y_coords / h).reshape(-1, 1)
    
    # Combine features
    features = np.hstack([hsv_features, x_norm, y_norm])
    return features
```

This gives 5 features per pixel: H, S, V, x, y

### Cross-Validation

Use k-fold cross-validation for better accuracy estimate:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## Troubleshooting

### Low Accuracy (<85%)

**Possible causes**:
- Insufficient training data (need more images)
- Poor labeling quality (check labels)
- Extreme lighting variations
- Similar skin and non-skin colors in dataset

**Solutions**:
1. Collect more diverse training images
2. Improve labeling accuracy
3. Add more non-skin examples
4. Try different model types

### Slow Inference (<15 FPS)

**Possible causes**:
- Large training set for KNN
- High image resolution
- Inefficient implementation

**Solutions**:
1. Reduce training set size (sample fewer examples)
2. Lower webcam resolution (320×240)
3. Use Naive Bayes or Decision Tree instead of KNN
4. Classify every 2nd or 3rd pixel

### High False Positives

**Problem**: Non-skin regions classified as skin

**Solutions**:
1. Add more non-skin labels (background, clothes, hair)
2. Use stricter threshold
3. Apply morphological opening to clean mask
4. Use Decision Tree (handles complex boundaries)

### High False Negatives

**Problem**: Skin regions not detected

**Solutions**:
1. Add more skin labels (various lighting)
2. Include shadow regions on face
3. Lower classification threshold
4. Use KNN with smaller k

### Unstable Bounding Box

**Problem**: Face bbox jumps around

**Solutions**:
1. Enable bbox smoothing (already implemented)
2. Increase smoothing factor (0.3 → 0.5)
3. Apply temporal filtering (Kalman filter)
4. Use larger connected component threshold

## Advanced Techniques

### Data Augmentation

Generate more training samples:

```python
from PIL import Image, ImageEnhance

def augment_image(image_path):
    """Apply random augmentations."""
    img = Image.open(image_path)
    
    # Brightness
    enhancer = ImageEnhance.Brightness(img)
    img_bright = enhancer.enhance(1.5)
    
    # Contrast
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(1.2)
    
    return [img, img_bright, img_contrast]
```

### Ensemble Methods

Combine multiple models:

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('nb', GaussianNB()),
        ('dt', DecisionTreeClassifier(max_depth=10))
    ],
    voting='soft'
)
```

### Active Learning

Iteratively improve model:

1. Train initial model
2. Run face tracker
3. Identify frames with poor detection
4. Label those frames manually
5. Retrain model
6. Repeat

## Saving and Loading Models

### Save Model
```python
import pickle

with open('models/my_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### Load Model
```python
import pickle

with open('models/my_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Model Versioning

Use descriptive filenames:

```
models/
├── skin_detector_knn_v1.pkl
├── skin_detector_knn_v2.pkl
├── skin_detector_nb_v1.pkl
└── metadata.json
```

metadata.json:
```json
{
  "skin_detector_knn_v2.pkl": {
    "model_type": "knn",
    "n_neighbors": 5,
    "train_date": "2025-11-01",
    "train_samples": 50000,
    "test_accuracy": 0.9423,
    "notes": "Trained with improved labeling"
  }
}
```

## Best Practices

1. **Collect diverse data**: Various lighting, angles, backgrounds
2. **Balance classes**: ~50% skin, ~50% non-skin samples
3. **Validate thoroughly**: Test in real conditions
4. **Version control**: Save models with metadata
5. **Iterate**: Continuously improve with new data

## Resources

- scikit-learn documentation: https://scikit-learn.org/
- HSV color space: https://en.wikipedia.org/wiki/HSL_and_HSV
- Skin detection papers: Search "skin detection HSV"

---

**Last Updated**: November 2025
