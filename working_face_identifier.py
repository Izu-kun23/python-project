#!/usr/bin/env python3
"""
Working Face Identifier - Uses OpenCV for face detection and basic recognition
This version works reliably without the problematic face_recognition library.
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from datetime import datetime


class WorkingFaceIdentifier:
    def __init__(self):
        """Initialize the working face identifier."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Database for known faces
        self.known_faces_dir = Path("famous_people")
        self.known_faces_dir.mkdir(exist_ok=True)
        
        # Face database file
        self.db_file = "face_database.pkl"
        self.face_database = self.load_face_database()
        
        # Create sample directories
        self.create_sample_directories()
    
    def create_sample_directories(self):
        """Create sample directories for famous people."""
        sample_people = [
            "elon_musk",
            "jeff_bezos", 
            "bill_gates",
            "steve_jobs",
            "mark_zuckerberg"
        ]
        
        for person in sample_people:
            person_dir = self.known_faces_dir / person
            person_dir.mkdir(exist_ok=True)
            
            readme_file = person_dir / "README.txt"
            if not readme_file.exists():
                with open(readme_file, 'w') as f:
                    f.write(f"Add images of {person.replace('_', ' ').title()} in this folder.\n")
                    f.write("Supported formats: .jpg, .jpeg, .png\n")
                    f.write("The more images you add, the better the recognition will be.\n")
    
    def load_face_database(self):
        """Load face database from file."""
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
        """Save face database to file."""
        with open(self.db_file, 'wb') as f:
            pickle.dump(self.face_database, f)
    
    def extract_face_features(self, face_roi):
        """Extract basic features from a face region."""
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        face_resized = cv2.resize(gray, (100, 100))
        
        # Extract basic features (histogram, edges, etc.)
        features = []
        
        # Histogram features
        hist = cv2.calcHist([face_resized], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
        
        # Edge features
        edges = cv2.Canny(face_resized, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Eye detection within face
        eyes = self.eye_cascade.detectMultiScale(face_resized)
        features.append(len(eyes))
        
        return np.array(features)
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors."""
        if len(features1) != len(features2):
            return 0.0
        
        # Normalize features
        features1 = features1 / (np.linalg.norm(features1) + 1e-8)
        features2 = features2 / (np.linalg.norm(features2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(features1, features2)
        return max(0.0, similarity)
    
    def add_person_to_database(self, person_name, image_paths):
        """Add a new person to the database."""
        print(f"Adding {person_name} to database...")
        
        person_dir = self.known_faces_dir / person_name.lower().replace(" ", "_")
        person_dir.mkdir(exist_ok=True)
        
        added_count = 0
        
        for i, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            # Load and process image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                print(f"No face found in {image_path}")
                continue
            
            # Use the first face found
            x, y, w, h = faces[0]
            face_roi = img[y:y+h, x:x+w]
            
            # Extract features
            features = self.extract_face_features(face_roi)
            
            # Save to database
            self.face_database['face_features'].append(features)
            self.face_database['face_names'].append(person_name)
            self.face_database['face_descriptions'].append(f"From {os.path.basename(image_path)}")
            
            # Save face image
            face_filename = person_dir / f"{person_name.lower().replace(' ', '_')}_{added_count + 1}.jpg"
            cv2.imwrite(str(face_filename), face_roi)
            
            added_count += 1
            print(f"  ✓ Added face from {os.path.basename(image_path)}")
        
        # Save database
        self.save_face_database()
        print(f"Added {added_count} faces for {person_name}")
    
    def identify_faces_in_image(self, image_path):
        """Identify faces in an image."""
        if not os.path.exists(image_path):
            return [], None
        
        img = cv2.imread(image_path)
        if img is None:
            return [], None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        
        for face in faces:
            x, y, w, h = face
            face_roi = img[y:y+h, x:x+w]
            
            # Extract features
            features = self.extract_face_features(face_roi)
            
            # Compare with known faces
            best_match = None
            best_similarity = 0.0
            
            for i, known_features in enumerate(self.face_database['face_features']):
                similarity = self.calculate_similarity(features, known_features)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = i
            
            if best_match is not None and best_similarity > 0.7:
                name = self.face_database['face_names'][best_match]
                confidence = best_similarity
            else:
                name = "Unknown"
                confidence = best_similarity
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': face
            })
        
        return results, img
    
    def identify_faces_in_video_stream(self, video_source=0):
        """Identify faces in a video stream."""
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
            
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for face in faces:
                x, y, w, h = [coord * 2 for coord in face]
                
                # Extract face features
                face_roi = frame[y:y+h, x:x+w]
                features = self.extract_face_features(face_roi)
                
                # Find best match
                best_match = None
                best_similarity = 0.0
                
                for i, known_features in enumerate(self.face_database['face_features']):
                    similarity = self.calculate_similarity(features, known_features)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = i
                
                if best_match is not None and best_similarity > 0.7:
                    name = self.face_database['face_names'][best_match]
                    confidence = best_similarity
                    color = (0, 255, 0)  # Green
                else:
                    name = "Unknown"
                    confidence = best_similarity
                    color = (0, 0, 255)  # Red
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{name} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y - 35), (x + w, y), color, cv2.FILLED)
                cv2.putText(frame, label, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Working Face Identifier', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"captured_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def show_database_stats(self):
        """Show statistics about the face database."""
        print(f"\n=== Face Database Statistics ===")
        print(f"Total faces in database: {len(self.face_database['face_names'])}")
        
        if self.face_database['face_names']:
            unique_names = set(self.face_database['face_names'])
            print(f"Unique people: {len(unique_names)}")
            for name in sorted(unique_names):
                count = self.face_database['face_names'].count(name)
                print(f"  - {name}: {count} faces")
        else:
            print("Database is empty. Add some famous people first!")


def main():
    """Main function."""
    identifier = WorkingFaceIdentifier()
    
    print("\n=== Working Face Identifier ===")
    print("This version uses OpenCV for reliable face detection and recognition!")
    
    while True:
        print("\nOptions:")
        print("1. Identify faces in an image")
        print("2. Live webcam identification")
        print("3. Add new famous person")
        print("4. Show database statistics")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            image_path = input("Enter path to image file: ").strip()
            results, img = identifier.identify_faces_in_image(image_path)
            
            if results:
                print(f"\nFound {len(results)} face(s):")
                for result in results:
                    print(f"  - {result['name']} (confidence: {result['confidence']:.2f})")
                
                # Show image with detected faces
                for result in results:
                    x, y, w, h = result['location']
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img, f"{result['name']} ({result['confidence']:.2f})", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                cv2.imshow('Detected Faces', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No faces detected in the image")
        
        elif choice == "2":
            print("Starting webcam stream...")
            identifier.identify_faces_in_video_stream()
        
        elif choice == "3":
            person_name = input("Enter famous person's name: ").strip()
            image_paths = input("Enter image paths (comma-separated): ").strip().split(",")
            image_paths = [path.strip() for path in image_paths]
            identifier.add_person_to_database(person_name, image_paths)
        
        elif choice == "4":
            identifier.show_database_stats()
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1-5.")


if __name__ == "__main__":
    main()
