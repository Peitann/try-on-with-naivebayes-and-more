# Quick Start Script for Try-On Filter
# This script guides you through the complete setup process

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Try-On Filter - Quick Start Setup" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Check directory structure
Write-Host ""
Write-Host "Verifying directory structure..." -ForegroundColor Yellow
$dirs = @("datasets/training_images", "datasets/labels", "models", "python_ml_tracking", "godot_project")
$allExist = $true
foreach ($dir in $dirs) {
    if (Test-Path $dir) {
        Write-Host "  ✓ $dir" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $dir not found" -ForegroundColor Red
        $allExist = $false
    }
}

if (-not $allExist) {
    Write-Host "Some directories are missing!" -ForegroundColor Red
    exit 1
}

# Guide user through workflow
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete! Next Steps:" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "STEP 1: Collect Training Data" -ForegroundColor Yellow
Write-Host "  python python_ml_tracking/data_collector.py" -ForegroundColor White
Write-Host "  → Capture 20-100 face images" -ForegroundColor Gray
Write-Host ""
Write-Host "STEP 2: Label Training Data" -ForegroundColor Yellow
Write-Host "  python python_ml_tracking/labeling_tool.py" -ForegroundColor White
Write-Host "  → Mark skin vs non-skin regions" -ForegroundColor Gray
Write-Host ""
Write-Host "STEP 3: Train ML Model" -ForegroundColor Yellow
Write-Host "  python python_ml_tracking/train_model.py" -ForegroundColor White
Write-Host "  → Train KNN, Naive Bayes, Decision Tree" -ForegroundColor Gray
Write-Host ""
Write-Host "STEP 4: Test Face Tracking" -ForegroundColor Yellow
Write-Host "  python python_ml_tracking/face_tracker.py" -ForegroundColor White
Write-Host "  → Verify tracking works" -ForegroundColor Gray
Write-Host ""
Write-Host "STEP 5: Run Complete System" -ForegroundColor Yellow
Write-Host "  Terminal 1: python python_ml_tracking/main.py" -ForegroundColor White
Write-Host "  Terminal 2: godot godot_project/project.godot" -ForegroundColor White
Write-Host "  → Start tracking server and Godot app" -ForegroundColor Gray
Write-Host ""
Write-Host "For detailed instructions, see README.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "Do you want to start data collection now? (y/n): " -ForegroundColor Yellow -NoNewline
$response = Read-Host

if ($response -eq "y" -or $response -eq "Y") {
    Write-Host ""
    Write-Host "Starting data collector..." -ForegroundColor Green
    python python_ml_tracking/data_collector.py
} else {
    Write-Host ""
    Write-Host "Setup complete! Run data collector when ready." -ForegroundColor Green
}
