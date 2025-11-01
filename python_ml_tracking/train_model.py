"""
ML Model Trainer for Skin Detection
Trains classical ML models (KNN, Naive Bayes, Decision Tree) using scikit-learn
"""

import numpy as np
import pickle
import os
import json
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, List
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_ml_tracking.image_utils import rgb_to_hsv


class SkinDetectionTrainer:
    """
    Trainer for classical ML-based skin detection.
    Supports KNN, Naive Bayes, and Decision Tree classifiers.
    """
    
    def __init__(self, model_type: str = "knn"):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of classifier ("knn", "naive_bayes", or "decision_tree")
        """
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        # Initialize model
        if model_type == "knn":
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == "naive_bayes":
            self.model = GaussianNB()
        elif model_type == "decision_tree":
            self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def extract_features_from_labels(self, image_dir: str, label_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract HSV features and labels from labeled images.
        
        Args:
            image_dir: Directory containing training images
            label_dir: Directory containing label JSON files
        
        Returns:
            Tuple of (features, labels)
            features: Array of HSV values, shape (n_samples, 3)
            labels: Array of labels (1 for skin, 0 for non-skin)
        """
        features_list = []
        labels_list = []
        
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        
        print(f"Extracting features from {len(label_files)} labeled images...")
        
        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)
            
            try:
                # Load label data
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
                
                image_path = label_data['image_path']
                
                # Handle relative paths
                if not os.path.isabs(image_path):
                    image_filename = os.path.basename(image_path)
                    image_path = os.path.join(image_dir, image_filename)
                
                # Load image
                image = Image.open(image_path)
                image_np = np.array(image)
                
                # Convert to HSV
                hsv_image = rgb_to_hsv(image_np)
                
                # Extract skin pixels
                skin_regions = label_data.get('skin_regions', [])
                for bbox in skin_regions:
                    x1, y1, x2, y2 = bbox
                    region_hsv = hsv_image[y1:y2, x1:x2, :]
                    
                    # Flatten and add to features
                    region_flat = region_hsv.reshape(-1, 3)
                    features_list.append(region_flat)
                    labels_list.append(np.ones(len(region_flat)))
                
                # Extract non-skin pixels
                non_skin_regions = label_data.get('non_skin_regions', [])
                for bbox in non_skin_regions:
                    x1, y1, x2, y2 = bbox
                    region_hsv = hsv_image[y1:y2, x1:x2, :]
                    
                    # Flatten and add to features
                    region_flat = region_hsv.reshape(-1, 3)
                    features_list.append(region_flat)
                    labels_list.append(np.zeros(len(region_flat)))
                
                print(f"  Processed: {os.path.basename(image_path)}")
            
            except Exception as e:
                print(f"  Error processing {label_file}: {e}")
        
        # Concatenate all features and labels
        X = np.vstack(features_list)
        y = np.hstack(labels_list)
        
        print(f"\nTotal samples: {len(X)}")
        print(f"  Skin pixels: {np.sum(y == 1)}")
        print(f"  Non-skin pixels: {np.sum(y == 0)}")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """
        Train the classifier.
        
        Args:
            X: Features array (n_samples, 3)
            y: Labels array (n_samples,)
            test_size: Fraction of data to use for testing
        """
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTraining {self.model_type} classifier...")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        print("Training complete!")
    
    def evaluate(self):
        """Evaluate the trained model on test set."""
        if self.model is None or self.X_test is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        
        print("\n" + "=" * 50)
        print("Model Evaluation")
        print("=" * 50)
        print(f"Model type: {self.model_type}")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        
        print("\nClassification Report (Test Set):")
        print(classification_report(self.y_test, y_pred_test, 
                                    target_names=['Non-Skin', 'Skin']))
        
        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(self.y_test, y_pred_test)
        print(cm)
        print("[[TN FP]")
        print(" [FN TP]]")
    
    def save_model(self, output_path: str):
        """
        Save trained model to disk.
        
        Args:
            output_path: Path to save model (pickle file)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\nModel saved to: {output_path}")
    
    @staticmethod
    def load_model(model_path: str):
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to model pickle file
        
        Returns:
            Loaded model
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model


def train_all_models(image_dir: str = "datasets/training_images",
                     label_dir: str = "datasets/labels",
                     output_dir: str = "models"):
    """
    Train all three model types and save them.
    
    Args:
        image_dir: Directory containing training images
        label_dir: Directory containing labels
        output_dir: Directory to save models
    """
    print("=" * 50)
    print("Training All Models")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    model_types = ["knn", "naive_bayes", "decision_tree"]
    
    for model_type in model_types:
        print(f"\n{'=' * 50}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'=' * 50}")
        
        try:
            # Create trainer
            trainer = SkinDetectionTrainer(model_type=model_type)
            
            # Extract features (only once, but repeated here for clarity)
            X, y = trainer.extract_features_from_labels(image_dir, label_dir)
            
            # Train
            trainer.train(X, y)
            
            # Evaluate
            trainer.evaluate()
            
            # Save
            model_path = os.path.join(output_dir, f"skin_detector_{model_type}.pkl")
            trainer.save_model(model_path)
        
        except Exception as e:
            print(f"Error training {model_type}: {e}")
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Models saved in: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train skin detection models")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="datasets/training_images",
        help="Directory containing training images"
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default="datasets/labels",
        help="Directory containing label JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["knn", "naive_bayes", "decision_tree"],
        choices=["knn", "naive_bayes", "decision_tree"],
        help="Model types to train (default: all three)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples for KNN (to reduce inference time with large datasets)"
    )
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.label_dir) or len(os.listdir(args.label_dir)) == 0:
        print("Error: No labels found!")
        print(f"Please create labels first using labeling_tool.py")
        print(f"Expected location: {args.label_dir}")
        sys.exit(1)
    elif not os.path.exists(args.image_dir):
        print("Error: Image directory not found!")
        print(f"Expected location: {args.image_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"Image directory: {args.image_dir}")
    print(f"Label directory: {args.label_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models to train: {', '.join(args.models)}")
    if args.max_samples:
        print(f"Max samples (KNN): {args.max_samples}")
    print("=" * 50)
    print()
    
    # Train each model
    for model_type in args.models:
        print(f"\n{'=' * 50}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'=' * 50}")
        
        try:
            # Create trainer
            trainer = SkinDetectionTrainer(model_type=model_type)
            
            # Extract features
            X, y = trainer.extract_features_from_labels(args.image_dir, args.label_dir)
            
            # Sample if needed (untuk KNN dengan dataset besar)
            if model_type == "knn" and args.max_samples and len(X) > args.max_samples:
                print(f"\n⚠️  Dataset terlalu besar untuk KNN!")
                print(f"   Original: {len(X):,} samples")
                print(f"   Sampling to: {args.max_samples:,} samples")
                print(f"   (untuk menjaga inference speed)")
                
                from sklearn.utils import resample
                X_sampled, y_sampled = resample(
                    X, y,
                    n_samples=args.max_samples,
                    stratify=y,
                    random_state=42
                )
                X, y = X_sampled, y_sampled
                print(f"   ✓ Sampling complete!")
            
            # Train
            trainer.train(X, y)
            
            # Evaluate
            trainer.evaluate()
            
            # Save
            model_path = os.path.join(args.output_dir, f"skin_detector_{model_type}.pkl")
            trainer.save_model(model_path)
        
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Models saved in: {args.output_dir}")
    print()
    print("Next steps:")
    print("1. Test face tracker: python python_ml_tracking/face_tracker.py")
    print("2. Run full system: python python_ml_tracking/main.py")
    print()
    print("💡 Tip: Untuk dataset besar (seperti CelebA), gunakan Naive Bayes atau Decision Tree")
    print("   untuk inference yang lebih cepat!")
