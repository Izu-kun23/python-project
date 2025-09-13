#!/usr/bin/env python3
"""
Simple Face Detector - Alternative version using only OpenCV
This version works even if face_recognition has issues.
"""

import cv2
import numpy as np
import os
from pathlib import Path


class SimpleFaceDetector:
    def __init__(self):
        """Initialize the simple face detector."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Known faces database (simple version)
        self.known_faces_dir = Path("known_faces")
        self.known_faces_dir.mkdir(exist_ok=True)
    
    def detect_faces_in_image(self, image_path):
        """Detect faces in an image."""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'location': (x, y, w, h),
                'confidence': 1.0  # OpenCV doesn't provide confidence scores
            })
        
        return results, img
    
    def detect_faces_in_video_stream(self, video_source=0):
        """Detect faces in a video stream (webcam)."""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Press 'q' to quit the video stream")
        print("Press 's' to save current frame")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Scale back up face locations
                x, y, w, h = [coord * 2 for coord in [x, y, w, h]]
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Try to detect eyes within the face
                roi_gray = gray[y//2:(y+h)//2, x//2:(x+w)//2]
                roi_color = frame[y:y+h, x:x+w]
                
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                
                # Add label
                cv2.putText(frame, f"Face #{len(faces)}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the resulting image
            cv2.imshow('Simple Face Detector', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame as {filename}")
        
        # Release the video capture and close windows
        cap.release()
        cv2.destroyAllWindows()
    
    def save_face_from_image(self, image_path, face_index=0, person_name="unknown"):
        """Save a detected face from an image."""
        results, img = self.detect_faces_in_image(image_path)
        
        if not results:
            print("No faces detected in the image")
            return False
        
        if face_index >= len(results):
            print(f"Face index {face_index} not found. Found {len(results)} faces.")
            return False
        
        # Get face coordinates
        x, y, w, h = results[face_index]['location']
        
        # Extract face
        face_img = img[y:y+h, x:x+w]
        
        # Create person directory
        person_dir = self.known_faces_dir / person_name.lower().replace(" ", "_")
        person_dir.mkdir(exist_ok=True)
        
        # Save face
        face_filename = person_dir / f"{person_name.lower().replace(' ', '_')}_{len(list(person_dir.glob('*.jpg')))}.jpg"
        cv2.imwrite(str(face_filename), face_img)
        
        print(f"Saved face as {face_filename}")
        return True
    
    def create_face_dataset(self, images_dir):
        """Create a dataset from images in a directory."""
        images_path = Path(images_dir)
        if not images_path.exists():
            print(f"Directory not found: {images_dir}")
            return
        
        for image_file in images_path.glob("*.{jpg,jpeg,png}"):
            print(f"Processing {image_file.name}...")
            
            # Extract filename without extension as person name
            person_name = image_file.stem
            
            # Try to save the first face found
            if self.save_face_from_image(str(image_file), 0, person_name):
                print(f"✓ Saved face from {image_file.name}")
            else:
                print(f"✗ No face found in {image_file.name}")


def main():
    """Main function to demonstrate the simple face detector."""
    detector = SimpleFaceDetector()
    
    print("\n=== Simple Face Detector ===")
    print("1. Detect faces in an image file")
    print("2. Detect faces in webcam stream")
    print("3. Save face from image")
    print("4. Create face dataset from directory")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            image_path = input("Enter path to image file: ").strip()
            results, img = detector.detect_faces_in_image(image_path)
            print(f"\nFound {len(results)} face(s)")
            
            if results:
                # Show the image with detected faces
                for i, result in enumerate(results):
                    x, y, w, h = result['location']
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img, f"Face {i+1}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                cv2.imshow('Detected Faces', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        elif choice == "2":
            print("Starting webcam stream...")
            detector.detect_faces_in_video_stream()
        
        elif choice == "3":
            image_path = input("Enter path to image file: ").strip()
            person_name = input("Enter person's name: ").strip()
            face_index = int(input("Enter face index (0 for first face): ") or "0")
            detector.save_face_from_image(image_path, face_index, person_name)
        
        elif choice == "4":
            images_dir = input("Enter path to directory with images: ").strip()
            detector.create_face_dataset(images_dir)
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1-5.")


if __name__ == "__main__":
    main()
