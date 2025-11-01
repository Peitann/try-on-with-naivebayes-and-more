"""
Installation verification script
Checks if all components are properly installed
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version OK (3.8+)")
        return True
    else:
        print("✗ Python version too old (need 3.8+)")
        return False


def check_dependencies():
    """Check required Python packages."""
    required = [
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('sklearn', 'scikit-learn'),
        ('pygame', 'pygame')
    ]
    
    all_ok = True
    for module, name in required:
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_ok = False
    
    return all_ok


def check_directories():
    """Check required directories exist."""
    required_dirs = [
        'python_ml_tracking',
        'godot_project',
        'datasets/training_images',
        'datasets/labels',
        'models',
        'docs'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} NOT found")
            all_ok = False
    
    return all_ok


def check_python_modules():
    """Check Python module files exist."""
    required_files = [
        'python_ml_tracking/__init__.py',
        'python_ml_tracking/image_utils.py',
        'python_ml_tracking/connected_components.py',
        'python_ml_tracking/data_collector.py',
        'python_ml_tracking/labeling_tool.py',
        'python_ml_tracking/train_model.py',
        'python_ml_tracking/webcam_capture.py',
        'python_ml_tracking/face_tracker.py',
        'python_ml_tracking/communication.py',
        'python_ml_tracking/main.py'
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ {os.path.basename(file_path)}")
        else:
            print(f"✗ {os.path.basename(file_path)} NOT found")
            all_ok = False
    
    return all_ok


def check_godot_files():
    """Check Godot project files exist."""
    required_files = [
        'godot_project/project.godot',
        'godot_project/scripts/MainScene.gd',
        'godot_project/scripts/FilterOverlay.gd',
        'godot_project/scripts/CanvasEditor.gd',
        'godot_project/scenes/MainScene.tscn'
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ {os.path.basename(file_path)}")
        else:
            print(f"✗ {os.path.basename(file_path)} NOT found")
            all_ok = False
    
    return all_ok


def check_documentation():
    """Check documentation files exist."""
    required_files = [
        'README.md',
        'docs/ARCHITECTURE.md',
        'docs/TRAINING.md',
        'docs/API.md',
        'docs/WORKFLOW_DIAGRAMS.md',
        'requirements.txt'
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ {os.path.basename(file_path)}")
        else:
            print(f"✗ {os.path.basename(file_path)} NOT found")
            all_ok = False
    
    return all_ok


def main():
    """Run all checks."""
    print("=" * 60)
    print("Try-On Filter - Installation Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Dependencies", check_dependencies),
        ("Directory Structure", check_directories),
        ("Python Modules", check_python_modules),
        ("Godot Files", check_godot_files),
        ("Documentation", check_documentation)
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All checks passed!")
        print("\nNext steps:")
        print("1. Run: python python_ml_tracking/data_collector.py")
        print("2. Run: python python_ml_tracking/labeling_tool.py")
        print("3. Run: python python_ml_tracking/train_model.py")
        print("4. Run: python python_ml_tracking/main.py")
        return 0
    else:
        print("\n✗ Some checks failed!")
        print("\nTo fix:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Verify all files are present")
        print("3. Re-run this script")
        return 1


if __name__ == "__main__":
    sys.exit(main())
