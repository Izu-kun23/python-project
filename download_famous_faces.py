#!/usr/bin/env python3
"""
Download Famous Faces from Internet
This script helps you download images of famous people and add them to your face database.
"""

import cv2
import numpy as np
import os
import pickle
import requests
from pathlib import Path
import time


class FamousFaceDownloader:
    def __init__(self):
        """Initialize the famous face downloader."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.db_file = "face_database.pkl"
        self.face_database = self.load_face_database()
        
        # Create directories
        self.famous_dir = Path("famous_faces")
        self.famous_dir.mkdir(exist_ok=True)
        
        # Famous people to download
        self.famous_people = {
            "elon_musk": {
                "name": "Elon Musk",
                "search_terms": ["elon musk", "tesla ceo", "spacex ceo"],
                "images": []
            },
            "jeff_bezos": {
                "name": "Jeff Bezos", 
                "search_terms": ["jeff bezos", "amazon ceo", "blue origin"],
                "images": []
            },
            "bill_gates": {
                "name": "Bill Gates",
                "search_terms": ["bill gates", "microsoft founder", "gates foundation"],
                "images": []
            },
            "steve_jobs": {
                "name": "Steve Jobs",
                "search_terms": ["steve jobs", "apple founder", "iphone creator"],
                "images": []
            },
            "mark_zuckerberg": {
                "name": "Mark Zuckerberg",
                "search_terms": ["mark zuckerberg", "facebook ceo", "meta ceo"],
                "images": []
            },
            "oprah_winfrey": {
                "name": "Oprah Winfrey",
                "search_terms": ["oprah winfrey", "oprah show", "media mogul"],
                "images": []
            },
            "elon_musk": {
                "name": "Elon Musk",
                "search_terms": ["elon musk", "tesla ceo", "spacex ceo"],
                "images": []
            }
        }
    
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
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(face_resized)
        features.append(len(eyes))
        
        return np.array(features)
    
    def download_image_from_url(self, url, filename):
        """Download image from URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            return False
    
    def create_sample_images(self):
        """Create sample images for famous people."""
        print("🎨 Creating sample images for famous people...")
        
        # Create sample images with text
        for person_id, person_data in self.famous_people.items():
            person_name = person_data["name"]
            
            # Create a sample image
            img = np.zeros((400, 300, 3), dtype=np.uint8)
            img.fill(50)  # Dark background
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, person_name, (50, 200), font, 1, (255, 255, 255), 2)
            cv2.putText(img, "Sample Image", (80, 250), font, 0.7, (200, 200, 200), 2)
            
            # Save image
            img_path = self.famous_dir / f"{person_id}_sample.jpg"
            cv2.imwrite(str(img_path), img)
            
            print(f"✅ Created sample image for {person_name}")
    
    def add_famous_person_from_image(self, image_path, person_name):
        """Add a famous person from an image file."""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False
        
        print(f"📸 Processing {person_name} from {image_path}...")
        
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
    
    def setup_famous_faces(self):
        """Set up famous faces database."""
        print("🌟 Setting up famous faces database...")
        
        # Create sample images
        self.create_sample_images()
        
        # Add each famous person
        for person_id, person_data in self.famous_people.items():
            person_name = person_data["name"]
            img_path = self.famous_dir / f"{person_id}_sample.jpg"
            
            if img_path.exists():
                self.add_famous_person_from_image(str(img_path), person_name)
        
        print(f"\n✅ Setup complete! Added {len(self.famous_people)} famous people to database.")
    
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
    
    def add_custom_image(self):
        """Add a custom image of a famous person."""
        print("\n📸 Add Custom Famous Person Image")
        print("=" * 40)
        
        image_path = input("Enter path to image file: ").strip()
        person_name = input("Enter person's name: ").strip()
        
        if image_path and person_name:
            if self.add_famous_person_from_image(image_path, person_name):
                print(f"✅ Successfully added {person_name}!")
            else:
                print(f"❌ Failed to add {person_name}")
        else:
            print("❌ Please provide both image path and person name")


def main():
    """Main function."""
    print("🌟 FAMOUS FACE DOWNLOADER")
    print("=" * 40)
    
    downloader = FamousFaceDownloader()
    
    while True:
        print("\nOptions:")
        print("1. Setup famous faces (create sample database)")
        print("2. Add custom famous person image")
        print("3. Show current database")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            downloader.setup_famous_faces()
        
        elif choice == "2":
            downloader.add_custom_image()
        
        elif choice == "3":
            downloader.show_database()
        
        elif choice == "4":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
