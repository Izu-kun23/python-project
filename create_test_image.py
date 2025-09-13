#!/usr/bin/env python3
"""
Create a test image for face detection demonstration
"""

import cv2
import numpy as np


def create_test_image():
    """Create a simple test image with text."""
    # Create a blank image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img.fill(50)  # Dark gray background
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    cv2.putText(img, "Face Detection Test Image", (50, 80), font, 1, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(img, "1. Add photos to famous_people/ folders", (50, 130), font, 0.6, (200, 200, 200), 2)
    cv2.putText(img, "2. Run: python working_face_identifier.py", (50, 160), font, 0.6, (200, 200, 200), 2)
    cv2.putText(img, "3. Choose option 2 for webcam demo", (50, 190), font, 0.6, (200, 200, 200), 2)
    
    # Add a simple face-like shape (circle)
    center = (300, 250)
    radius = 80
    cv2.circle(img, center, radius, (255, 255, 255), 2)
    
    # Eyes
    cv2.circle(img, (center[0] - 25, center[1] - 20), 10, (255, 255, 255), -1)
    cv2.circle(img, (center[0] + 25, center[1] - 20), 10, (255, 255, 255), -1)
    
    # Nose
    cv2.line(img, (center[0], center[1] - 10), (center[0], center[1] + 10), (255, 255, 255), 2)
    
    # Mouth
    cv2.ellipse(img, (center[0], center[1] + 30), (20, 10), 0, 0, 180, (255, 255, 255), 2)
    
    # Save the image
    cv2.imwrite("test_image.jpg", img)
    print("Created test_image.jpg")
    
    return "test_image.jpg"


if __name__ == "__main__":
    create_test_image()
