#!/usr/bin/env python3
"""
Add People to Face Database
This script helps you add famous people or anyone to your face database.
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path


class FaceDatabaseManager:
    def __init__(self):
        """Initialize the face database manager."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.db_file = "face_database.pkl"
        self.face_database = self.load_face_database()
    
    def load_face_database(self):
        """Load face database."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        return {
            'face_features': [],
            'face_names': [],
            'face_descriptions': []
        }
    
    def save_face_database(self):
        """Save face database."""
        with open(self.db_file, 'wb') as f:
            pickle.dump(self.face_database, f)
    
    def extract_face_features(self, face_roi):
        """Extract features from face."""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(gray, (100, 100))
        
        features = []
        
        # Histogram features
        hist = cv2.calcHist([face_resized], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
        
        # Edge features
        edges = cv2.Canny(face_resized, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        return np.array(features)
    
    def add_person_from_image(self, image_path, person_name):
        """Add a person from an image file."""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False
        
        print(f"📸 Processing {image_path}...")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return False
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            print(f"❌ No faces found in {image_path}")
            return False
        
        print(f"✅ Found {len(faces)} face(s)")
        
        # Use the first face
        x, y, w, h = faces[0]
        face_roi = img[y:y+h, x:x+w]
        
        # Extract features
        features = self.extract_face_features(face_roi)
        
        # Add to database
        self.face_database['face_features'].append(features)
        self.face_database['face_names'].append(person_name)
        self.face_database['face_descriptions'].append(f"From {os.path.basename(image_path)}")
        
        # Save database
        self.save_face_database()
        
        print(f"✅ Added {person_name} to database!")
        return True
    
    def add_person_from_webcam(self, person_name):
        """Add a person by taking a photo with webcam."""
        print(f"📸 Taking photo of {person_name}...")
        
        # Find working camera
        camera_id = None
        for cam_id in [0, 1, 2]:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    camera_id = cam_id
                    cap.release()
                    break
            cap.release()
        
        if camera_id is None:
            print("❌ No camera available!")
            return False
        
        cap = cv2.VideoCapture(camera_id)
        
        print("📋 Instructions:")
        print("- Position your face in the camera")
        print("- Press SPACE to take photo")
        print("- Press 'q' to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangle around face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{person_name}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(frame, "SPACE=take photo, q=cancel", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f'Add {person_name}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space bar
                if len(faces) > 0:
                    # Take the photo
                    x, y, w, h = faces[0]
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Extract features
                    features = self.extract_face_features(face_roi)
                    
                    # Add to database
                    self.face_database['face_features'].append(features)
                    self.face_database['face_names'].append(person_name)
                    self.face_database['face_descriptions'].append("From webcam")
                    
                    # Save database
                    self.save_face_database()
                    
                    print(f"✅ Added {person_name} to database!")
                    
                    # Save the photo
                    photo_filename = f"{person_name.lower().replace(' ', '_')}_photo.jpg"
                    cv2.imwrite(photo_filename, face_roi)
                    print(f"📸 Saved photo as {photo_filename}")
                    
                    break
                else:
                    print("❌ No face detected! Position your face in the camera.")
            elif key == ord('q'):
                print("❌ Cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def show_database(self):
        """Show current database."""
        print(f"\n📊 Current Face Database:")
        print(f"Total faces: {len(self.face_database['face_names'])}")
        
        if self.face_database['face_names']:
            unique_names = set(self.face_database['face_names'])
            print(f"Unique people: {len(unique_names)}")
            for name in sorted(unique_names):
                count = self.face_database['face_names'].count(name)
                print(f"  - {name}: {count} faces")
        else:
            print("Database is empty")


def main():
    """Main function."""
    print("🎯 FACE DATABASE MANAGER")
    print("=" * 40)
    
    manager = FaceDatabaseManager()
    
    while True:
        print("\nOptions:")
        print("1. Add person from image file")
        print("2. Add person using webcam")
        print("3. Show database")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            image_path = input("Enter image file path: ").strip()
            person_name = input("Enter person's name: ").strip()
            if image_path and person_name:
                manager.add_person_from_image(image_path, person_name)
        
        elif choice == "2":
            person_name = input("Enter person's name: ").strip()
            if person_name:
                manager.add_person_from_webcam(person_name)
        
        elif choice == "3":
            manager.show_database()
        
        elif choice == "4":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
