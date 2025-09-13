#!/usr/bin/env python3
"""
Demo script for Famous Face Identifier
This script demonstrates the basic functionality without requiring user interaction.
"""

import cv2
import numpy as np
from famous_face_identifier import FamousFaceIdentifier


def create_sample_image():
    """Create a sample image with text for demonstration."""
    # Create a simple image with text
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img.fill(50)  # Dark gray background
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Famous Face Identifier Demo", (50, 100), font, 1, (255, 255, 255), 2)
    cv2.putText(img, "Add photos to famous_people/ folders", (50, 150), font, 0.7, (200, 200, 200), 2)
    cv2.putText(img, "Then run: python famous_face_identifier.py", (50, 200), font, 0.7, (200, 200, 200), 2)
    
    # Add some decorative elements
    cv2.rectangle(img, (50, 250), (550, 350), (100, 100, 255), 2)
    cv2.putText(img, "Press 'q' to start webcam demo", (100, 320), font, 0.8, (255, 255, 255), 2)
    
    return img


def demo_webcam():
    """Demonstrate webcam functionality."""
    print("Starting webcam demo...")
    print("Press 'q' to quit")
    
    # Initialize the identifier
    identifier = FamousFaceIdentifier()
    
    # Check if we have any famous people in the database
    if len(identifier.known_face_names) == 0:
        print("\n⚠️  No famous people found in database!")
        print("Please add some photos to the famous_people/ folders first.")
        print("Run 'python setup.py' to create the folder structure.")
        return
    
    print(f"Found {len(identifier.known_face_names)} famous people in database:")
    for name in set(identifier.known_face_names):
        print(f"  - {name}")
    
    # Start webcam identification
    identifier.identify_faces_in_video_stream()


def demo_image_processing():
    """Demonstrate image processing functionality."""
    print("Image processing demo...")
    
    # Create a sample image
    sample_img = create_sample_image()
    sample_path = "sample_image.jpg"
    cv2.imwrite(sample_path, sample_img)
    
    # Initialize the identifier
    identifier = FamousFaceIdentifier()
    
    # Try to identify faces in the sample image
    results = identifier.identify_faces_in_image(sample_path)
    
    print(f"Analysis of {sample_path}:")
    if results:
        for result in results:
            print(f"  Found: {result['name']} (confidence: {result['confidence']:.2f})")
    else:
        print("  No faces detected in the sample image")
    
    # Clean up
    import os
    if os.path.exists(sample_path):
        os.remove(sample_path)


def main():
    """Main demo function."""
    print("=== Famous Face Identifier Demo ===\n")
    
    # Demo 1: Image processing
    demo_image_processing()
    
    print("\n" + "="*50 + "\n")
    
    # Demo 2: Webcam (optional)
    webcam_choice = input("Would you like to test the webcam functionality? (y/n): ").lower()
    if webcam_choice == 'y':
        demo_webcam()
    
    print("\nDemo complete!")
    print("\nTo use the full application:")
    print("1. Run: python famous_face_identifier.py")
    print("2. Add photos of famous people to the famous_people/ folders")
    print("3. Choose option 2 for live webcam identification")


if __name__ == "__main__":
    main()
