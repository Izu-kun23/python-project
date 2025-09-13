#!/usr/bin/env python3
"""
Add Famous People to Face Database
This script helps you add real famous people to your face database.
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path


class FamousPeopleManager:
    def __init__(self):
        """Initialize the famous people manager."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.db_file = "face_database.pkl"
        self.face_database = self.load_face_database()
        
        # Create directories
        self.famous_dir = Path("famous_faces")
        self.famous_dir.mkdir(exist_ok=True)
        
        # Famous people list
        self.famous_people = [
            "Elon Musk",
            "Jeff Bezos", 
            "Bill Gates",
            "Steve Jobs",
            "Mark Zuckerberg",
            "Oprah Winfrey",
            "Barack Obama",
            "Donald Trump",
            "Taylor Swift",
            "Tom Hanks",
            "Jennifer Lawrence",
            "Leonardo DiCaprio",
            "Emma Watson",
            "Ryan Reynolds",
            "Scarlett Johansson"
        ]
    
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
            print("💡 Make sure the image contains a clear face")
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
    
    def create_dummy_faces(self):
        """Create dummy faces for testing (so you can see names in live detection)."""
        print("🎨 Creating dummy faces for testing...")
        
        for person_name in self.famous_people:
            # Create dummy features
            dummy_features = np.random.random(34)  # 32 histogram + 1 edge + 1 eye
            
            # Add to database
            self.face_database['face_features'].append(dummy_features)
            self.face_database['face_names'].append(person_name)
            self.face_database['face_descriptions'].append("Dummy face for testing")
        
        # Save database
        self.save_face_database()
        
        print(f"✅ Created {len(self.famous_people)} dummy faces for testing!")
        print("💡 These will show names in live detection, but won't recognize real faces")
        print("   Use option 2 to add real photos for actual recognition")
    
    def add_real_famous_image(self):
        """Add a real famous person image."""
        print("\n📸 Add Real Famous Person Image")
        print("=" * 40)
        
        print("📋 Instructions:")
        print("1. Download a photo of a famous person from the internet")
        print("2. Save it to your computer")
        print("3. Enter the path to the image below")
        
        print(f"\n📁 You can save images in: {self.famous_dir.absolute()}")
        
        image_path = input("\nEnter path to image file: ").strip()
        person_name = input("Enter person's name: ").strip()
        
        if image_path and person_name:
            if self.add_famous_person_from_image(image_path, person_name):
                print(f"✅ Successfully added {person_name}!")
                print("🎯 Now when you run the live face identifier, it should recognize this person!")
            else:
                print(f"❌ Failed to add {person_name}")
        else:
            print("❌ Please provide both image path and person name")
    
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
    
    def show_instructions(self):
        """Show detailed instructions."""
        print("\n📖 HOW TO ADD REAL FAMOUS PEOPLE")
        print("=" * 50)
        
        print("\n1. 🌐 Find photos online:")
        print("   - Go to Google Images")
        print("   - Search for 'Elon Musk photo' or any famous person")
        print("   - Look for clear, front-facing photos")
        print("   - Download high-quality images")
        
        print("\n2. 💾 Save the images:")
        print(f"   - Save them in: {self.famous_dir.absolute()}")
        print("   - Use descriptive names like 'elon_musk.jpg'")
        
        print("\n3. ➕ Add to database:")
        print("   - Use option 2 in this menu")
        print("   - Enter the image path and person's name")
        
        print("\n4. 🎯 Test recognition:")
        print("   - Run: python live_face_identifier.py")
        print("   - Show the person's photo to the camera")
        print("   - It should recognize and show their name!")
        
        print("\n💡 Tips for better recognition:")
        print("   - Use clear, well-lit photos")
        print("   - Front-facing photos work best")
        print("   - Multiple photos of the same person improve accuracy")
        
        print(f"\n📁 Your famous_faces folder: {self.famous_dir.absolute()}")
    
    def quick_setup(self):
        """Quick setup with dummy faces."""
        print("🚀 QUICK SETUP")
        print("=" * 20)
        print("This will create dummy faces so you can see names in live detection.")
        print("For real recognition, add actual photos using option 2.")
        
        confirm = input("\nContinue? (y/n): ").lower()
        if confirm == 'y':
            self.create_dummy_faces()
        else:
            print("Setup cancelled.")


def main():
    """Main function."""
    print("🌟 FAMOUS PEOPLE MANAGER")
    print("=" * 40)
    
    manager = FamousPeopleManager()
    
    while True:
        print("\nOptions:")
        print("1. Quick setup (create dummy faces for testing)")
        print("2. Add real famous person image")
        print("3. Show current database")
        print("4. Show instructions")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            manager.quick_setup()
        
        elif choice == "2":
            manager.add_real_famous_image()
        
        elif choice == "3":
            manager.show_database()
        
        elif choice == "4":
            manager.show_instructions()
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
