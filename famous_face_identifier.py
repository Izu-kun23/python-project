#!/usr/bin/env python3
"""
Famous Person Face Identifier
A Python application that identifies famous people in images or video streams.
"""

import cv2
import face_recognition
import numpy as np
import os
from pathlib import Path
import pickle
import json


class FamousFaceIdentifier:
    def __init__(self):
        """Initialize the famous face identifier."""
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load or create the famous people database
        self.load_famous_faces()
    
    def load_famous_faces(self):
        """Load famous face encodings from the database."""
        db_path = "famous_faces_db.pkl"
        
        if os.path.exists(db_path):
            print("Loading existing famous faces database...")
            with open(db_path, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"Loaded {len(self.known_face_names)} famous people from database")
        else:
            print("No existing database found. Creating new one...")
            self.create_famous_faces_database()
    
    def create_famous_faces_database(self):
        """Create a database of famous faces from images in the famous_people folder."""
        famous_people_dir = Path("famous_people")
        famous_people_dir.mkdir(exist_ok=True)
        
        # Create sample famous people structure
        sample_people = {
            "elon_musk": "Elon Musk",
            "jeff_bezos": "Jeff Bezos", 
            "bill_gates": "Bill Gates",
            "steve_jobs": "Steve Jobs",
            "mark_zuckerberg": "Mark Zuckerberg"
        }
        
        # Create subdirectories for each famous person
        for folder_name, display_name in sample_people.items():
            person_dir = famous_people_dir / folder_name
            person_dir.mkdir(exist_ok=True)
            
            # Create a README file explaining how to add images
            readme_path = person_dir / "README.txt"
            if not readme_path.exists():
                with open(readme_path, 'w') as f:
                    f.write(f"Add images of {display_name} in this folder.\n")
                    f.write("Supported formats: .jpg, .jpeg, .png\n")
                    f.write("The more images you add, the better the recognition will be.\n")
        
        # Load any existing images
        self.load_images_from_directory(famous_people_dir)
        
        # Save the database
        self.save_famous_faces_database()
    
    def load_images_from_directory(self, directory):
        """Load face encodings from images in the directory."""
        for person_dir in directory.iterdir():
            if person_dir.is_dir() and person_dir.name != "__pycache__":
                person_name = person_dir.name.replace("_", " ").title()
                print(f"Loading images for {person_name}...")
                
                image_count = 0
                for image_file in person_dir.glob("*.{jpg,jpeg,png}"):
                    try:
                        # Load the image
                        image = face_recognition.load_image_file(str(image_file))
                        
                        # Find face encodings
                        face_encodings = face_recognition.face_encodings(image)
                        
                        if face_encodings:
                            # Use the first face found
                            self.known_face_encodings.append(face_encodings[0])
                            self.known_face_names.append(person_name)
                            image_count += 1
                            print(f"  ✓ Loaded {image_file.name}")
                        else:
                            print(f"  ✗ No face found in {image_file.name}")
                            
                    except Exception as e:
                        print(f"  ✗ Error loading {image_file.name}: {e}")
                
                print(f"Loaded {image_count} images for {person_name}")
    
    def save_famous_faces_database(self):
        """Save the famous faces database to disk."""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        
        with open("famous_faces_db.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved database with {len(self.known_face_names)} famous people")
    
    def identify_faces_in_image(self, image_path):
        """Identify famous people in a single image."""
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        results = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                name = self.known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
            else:
                name = "Unknown"
                confidence = 0
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': face_location
            })
        
        return results
    
    def identify_faces_in_video_stream(self, video_source=0):
        """Identify famous people in a video stream (webcam)."""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Press 'q' to quit the video stream")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Process each face
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Scale back up face locations
                top, right, bottom, left = [coord * 4 for coord in face_location]
                
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                # Find the best match
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    color = (0, 255, 0)  # Green for known faces
                else:
                    name = "Unknown"
                    confidence = 0
                    color = (0, 0, 255)  # Red for unknown faces
                
                # Draw rectangle around face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label
                label = f"{name} ({confidence:.2f})"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the resulting image
            cv2.imshow('Famous Face Identifier', frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the video capture and close windows
        cap.release()
        cv2.destroyAllWindows()
    
    def add_new_famous_person(self, person_name, image_paths):
        """Add a new famous person to the database."""
        person_name_clean = person_name.lower().replace(" ", "_")
        person_dir = Path("famous_people") / person_name_clean
        person_dir.mkdir(exist_ok=True)
        
        # Copy images to the person's directory
        for i, image_path in enumerate(image_paths):
            if os.path.exists(image_path):
                extension = Path(image_path).suffix
                new_path = person_dir / f"{person_name_clean}_{i+1}{extension}"
                
                # Load and process the image
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    # Save the processed image
                    cv2.imwrite(str(new_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    
                    # Add to database
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(person_name)
        
        # Save updated database
        self.save_famous_faces_database()
        print(f"Added {person_name} to the famous faces database")


def main():
    """Main function to demonstrate the famous face identifier."""
    identifier = FamousFaceIdentifier()
    
    print("\n=== Famous Face Identifier ===")
    print("1. Identify faces in an image file")
    print("2. Identify faces in webcam stream")
    print("3. Add new famous person")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            image_path = input("Enter path to image file: ").strip()
            if os.path.exists(image_path):
                results = identifier.identify_faces_in_image(image_path)
                print(f"\nFound {len(results)} face(s):")
                for result in results:
                    print(f"  - {result['name']} (confidence: {result['confidence']:.2f})")
            else:
                print("Image file not found!")
        
        elif choice == "2":
            print("Starting webcam stream...")
            identifier.identify_faces_in_video_stream()
        
        elif choice == "3":
            person_name = input("Enter famous person's name: ").strip()
            image_paths = input("Enter image paths (comma-separated): ").strip().split(",")
            image_paths = [path.strip() for path in image_paths]
            identifier.add_new_famous_person(person_name, image_paths)
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1-4.")


if __name__ == "__main__":
    main()
