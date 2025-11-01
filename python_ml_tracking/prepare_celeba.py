"""
Preprocessing CelebA dataset untuk skin detection training.
Membuat labels otomatis dengan heuristic sederhana.
"""

import os
import numpy as np
from PIL import Image
import json
import random
import argparse


def prepare_celeba_labels(celeba_dir: str, output_label_dir: str, 
                          num_samples: int = 1000, seed: int = 42):
    """
    Buat labels otomatis untuk CelebA dataset.
    
    Args:
        celeba_dir: Path ke folder CelebA images
        output_label_dir: Path output untuk labels
        num_samples: Jumlah images yang akan diproses
        seed: Random seed untuk reproducibility
    """
    os.makedirs(output_label_dir, exist_ok=True)
    random.seed(seed)
    
    # Get list of images
    print(f"Scanning {celeba_dir}...")
    image_files = [f for f in os.listdir(celeba_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"Error: No images found in {celeba_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Sample random images
    if num_samples < len(image_files):
        image_files = random.sample(image_files, num_samples)
        print(f"Randomly selected {num_samples} images")
    
    print(f"\nProcessing {len(image_files)} images...")
    
    success_count = 0
    error_count = 0
    
    for idx, img_file in enumerate(image_files):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)} images...")
        
        img_path = os.path.join(celeba_dir, img_file)
        
        try:
            # Load image
            img = Image.open(img_path)
            w, h = img.size
            
            # CelebA images are aligned and centered on faces
            # Face region: center 60% of image
            center_x, center_y = w // 2, h // 2
            face_w, face_h = int(w * 0.6), int(h * 0.6)
            
            face_bbox = (
                center_x - face_w // 2,
                center_y - face_h // 2,
                center_x + face_w // 2,
                center_y + face_h // 2
            )
            
            # Background regions (corners)
            bg_size = min(30, w // 10, h // 10)
            background_bboxes = [
                (0, 0, bg_size, bg_size),                      # Top-left
                (w - bg_size, 0, w, bg_size),                  # Top-right
                (0, h - bg_size, bg_size, h),                  # Bottom-left
                (w - bg_size, h - bg_size, w, h)               # Bottom-right
            ]
            
            # Save label
            label_data = {
                "image_path": img_path,
                "image_size": [w, h],
                "skin_regions": [face_bbox],
                "non_skin_regions": background_bboxes
            }
            
            label_filename = os.path.splitext(img_file)[0] + ".json"
            label_path = os.path.join(output_label_dir, label_filename)
            
            with open(label_path, 'w') as f:
                json.dump(label_data, f, indent=2)
            
            success_count += 1
        
        except Exception as e:
            print(f"\nError processing {img_file}: {e}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Success: {success_count} images")
    print(f"  Errors: {error_count} images")
    print(f"  Labels saved to: {output_label_dir}")
    print(f"{'='*60}")


def verify_labels(label_dir: str, num_check: int = 5):
    """
    Verify beberapa label files untuk quality check.
    
    Args:
        label_dir: Directory dengan label files
        num_check: Jumlah files yang akan dicek
    """
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    
    if not label_files:
        print(f"No label files found in {label_dir}")
        return
    
    print(f"\nVerifying {num_check} random labels...")
    check_files = random.sample(label_files, min(num_check, len(label_files)))
    
    for label_file in check_files:
        label_path = os.path.join(label_dir, label_file)
        
        with open(label_path, 'r') as f:
            data = json.load(f)
        
        print(f"\n{label_file}:")
        print(f"  Image: {os.path.basename(data['image_path'])}")
        print(f"  Size: {data['image_size']}")
        print(f"  Skin regions: {len(data['skin_regions'])}")
        print(f"  Non-skin regions: {len(data['non_skin_regions'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CelebA dataset for skin detection training"
    )
    parser.add_argument(
        "--celeba-dir",
        type=str,
        required=True,
        help="Path to CelebA image directory (e.g., D:/datasets/CelebA/img_align_celeba)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/labels_celeba",
        help="Output directory for labels (default: datasets/labels_celeba)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of images to process (default: 1000)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify labels after creation"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("CelebA Dataset Preparation")
    print("="*60)
    print(f"Input directory: {args.celeba_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print("="*60)
    print()
    
    # Check if input directory exists
    if not os.path.exists(args.celeba_dir):
        print(f"Error: Directory not found: {args.celeba_dir}")
        print("\nPlease download CelebA dataset from:")
        print("  http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("  or https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
        return 1
    
    # Prepare labels
    prepare_celeba_labels(
        args.celeba_dir,
        args.output_dir,
        args.num_samples
    )
    
    # Verify if requested
    if args.verify:
        verify_labels(args.output_dir)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("1. Train model with Naive Bayes (recommended for large datasets):")
    print(f"   python python_ml_tracking/train_model.py \\")
    print(f"       --image-dir \"{args.celeba_dir}\" \\")
    print(f"       --label-dir \"{args.output_dir}\" \\")
    print(f"       --models naive_bayes decision_tree")
    print()
    print("2. Test the model:")
    print("   python python_ml_tracking/face_tracker.py")
    print()
    print("3. Run the complete system:")
    print("   python python_ml_tracking/main.py --model models/skin_detector_naive_bayes.pkl")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
