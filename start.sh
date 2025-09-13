#!/bin/bash

# Face Identifier Start Script
echo "🎯 Face Identifier - Starting..."

# Check if virtual environment exists
if [ ! -d "face_identifier_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv face_identifier_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source face_identifier_env/bin/activate

# Check if packages are installed
echo "Checking dependencies..."
python -c "import cv2, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install opencv-python numpy Pillow imutils
fi

# Run the quick start menu
echo "Starting Face Identifier..."
python quick_start.py
