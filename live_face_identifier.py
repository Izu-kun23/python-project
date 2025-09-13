#!/usr/bin/env python3
"""
Live Face Identifier - Real-time face detection and recognition
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path


class LiveFaceIdentifier:
    def __init__(self):
        """Initialize the live face identifier."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Find working camera
        self.camera_id = self.find_working_camera()
        
        # Face database
        self.db_file = "face_database.pkl"
        self.face_database = self.load_face_database()
        
        # Create sample faces if database is empty
        if not self.face_database['face_names']:
            self.create_sample_faces()
    
    def find_working_camera(self):
        """Find a working camera."""
        print("🎥 Finding working camera...")
        
        for camera_id in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    print(f"✅ Found working camera: {camera_id}")
                    return camera_id
            cap.release()
        
        print("❌ No working camera found!")
        return None
    
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
        
        # Eye detection
        eyes = self.eye_cascade.detectMultiScale(face_resized)
        features.append(len(eyes))
        
        return np.array(features)
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between features."""
        if len(features1) != len(features2):
            return 0.0
        
        features1 = features1 / (np.linalg.norm(features1) + 1e-8)
        features2 = features2 / (np.linalg.norm(features2) + 1e-8)
        
        similarity = np.dot(features1, features2)
        return max(0.0, similarity)
    
    def create_sample_faces(self):
        """Create some sample faces for testing."""
        print("📝 Creating sample faces for testing...")
        
        # Create sample data
        sample_names = ["Test Person", "Demo Face", "Sample User"]
        
        for i, name in enumerate(sample_names):
            # Create dummy features
            dummy_features = np.random.random(34)  # 32 histogram + 1 edge + 1 eye
            self.face_database['face_features'].append(dummy_features)
            self.face_database['face_names'].append(name)
            self.face_database['face_descriptions'].append(f"Sample face {i+1}")
        
        self.save_face_database()
        print(f"✅ Created {len(sample_names)} sample faces")
        print("💡 Add real photos to improve recognition!")
    
    def add_current_face(self, face_roi, person_name):
        """Add current face to database."""
        features = self.extract_face_features(face_roi)
        
        self.face_database['face_features'].append(features)
        self.face_database['face_names'].append(person_name)
        self.face_database['face_descriptions'].append(f"Added from live camera")
        
        self.save_face_database()
        print(f"✅ Added {person_name} to database!")
    
    def identify_face(self, face_roi):
        """Identify a face."""
        features = self.extract_face_features(face_roi)
        
        best_match = None
        best_similarity = 0.0
        
        for i, known_features in enumerate(self.face_database['face_features']):
            similarity = self.calculate_similarity(features, known_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = i
        
        if best_match is not None and best_similarity > 0.7:
            return self.face_database['face_names'][best_match], best_similarity
        else:
            return "Unknown", best_similarity
    
    def run_live_detection(self):
        """Run live face detection and recognition."""
        if self.camera_id is None:
            print("❌ No camera available!")
            return
        
        print(f"\n🎥 Starting live face identification with camera {self.camera_id}")
        print("📋 Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'a' to add current face to database")
        print("  - Press 'r' to reset database")
        
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("\n🎬 Live detection started! Look at the camera window...")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Resize for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # Process each face
                for face in faces:
                    x, y, w, h = [coord * 2 for coord in face]  # Scale back up
                    
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Identify face
                    name, confidence = self.identify_face(face_roi)
                    
                    # Choose color based on recognition
                    if name != "Unknown":
                        color = (0, 255, 0)  # Green for known
                        label = f"{name} ({confidence:.2f})"
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        label = f"Unknown ({confidence:.2f})"
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y - 35), (x + w, y), color, cv2.FILLED)
                    cv2.putText(frame, label, (x + 6, y - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Add status info
                cv2.putText(frame, f"Faces: {len(faces)} | Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "q=quit, s=save, a=add face, r=reset", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Live Face Identifier', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting live detection...")
                    break
                elif key == ord('s'):
                    filename = f"live_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved frame as {filename}")
                elif key == ord('a') and len(faces) > 0:
                    # Add first detected face
                    x, y, w, h = [coord * 2 for coord in faces[0]]
                    face_roi = frame[y:y+h, x:x+w]
                    person_name = input("Enter person's name: ").strip()
                    if person_name:
                        self.add_current_face(face_roi, person_name)
                elif key == ord('r'):
                    confirm = input("Reset database? (y/n): ").lower()
                    if confirm == 'y':
                        self.face_database = {
                            'face_features': [],
                            'face_names': [],
                            'face_descriptions': []
                        }
                        self.save_face_database()
                        print("🗑️ Database reset!")
        
        except KeyboardInterrupt:
            print("\n👋 Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Live detection finished!")
    
    def show_database_info(self):
        """Show database information."""
        print(f"\n📊 Face Database Info:")
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
    print("🎯 LIVE FACE IDENTIFIER")
    print("=" * 40)
    
    identifier = LiveFaceIdentifier()
    
    if identifier.camera_id is None:
        print("❌ No camera available. Please check your camera connection.")
        return
    
    identifier.show_database_info()
    
    print("\n🚀 Starting live face identification...")
    identifier.run_live_detection()


if __name__ == "__main__":
    main()
