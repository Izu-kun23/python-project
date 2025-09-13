#!/usr/bin/env python3
"""
Setup Famous Faces Database
This script creates a database with famous people that you can recognize by name.
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path


class FamousFaceSetup:
    def __init__(self):
        """Initialize the famous face setup."""
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
            "Tom Hanks"
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
    
    def create_sample_faces(self):
        """Create sample faces for famous people."""
        print("🎨 Creating sample faces for famous people...")
        
        for person_name in self.famous_people:
            # Create a sample image with the person's name
            img = np.zeros((400, 300, 3), dtype=np.uint8)
            img.fill(50)  # Dark background
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Split name if too long
            name_parts = person_name.split()
            if len(person_name) > 15:
                # Put first name on top, last name on bottom
                cv2.putText(img, name_parts[0], (50, 180), font, 1, (255, 255, 255), 2)
                if len(name_parts) > 1:
                    cv2.putText(img, name_parts[1], (50, 220), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(img, person_name, (50, 200), font, 1, (255, 255, 255), 2)
            
            cv2.putText(img, "Sample", (100, 250), font, 0.6, (200, 200, 200), 2)
            
            # Save image
            safe_name = person_name.lower().replace(" ", "_")
            img_path = self.famous_dir / f"{safe_name}_sample.jpg"
            cv2.imwrite(str(img_path), img)
            
            print(f"✅ Created sample for {person_name}")
    
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
        self.create_sample_faces()
        
        # Add each famous person
        added_count = 0
        for person_name in self.famous_people:
            safe_name = person_name.lower().replace(" ", "_")
            img_path = self.famous_dir / f"{safe_name}_sample.jpg"
            
            if img_path.exists():
                if self.add_famous_person_from_image(str(img_path), person_name):
                    added_count += 1
        
        print(f"\n✅ Setup complete! Added {added_count} famous people to database.")
        print("\n💡 To improve recognition:")
        print("1. Download real photos of these famous people from the internet")
        print("2. Save them in the famous_faces/ folder")
        print("3. Use option 2 to add them to the database")
    
    def add_real_famous_image(self):
        """Add a real famous person image."""
        print("\n📸 Add Real Famous Person Image")
        print("=" * 40)
        print("Instructions:")
        print("1. Download a photo of a famous person from the internet")
        print("2. Save it to your computer")
        print("3. Enter the path to the image below")
        
        image_path = input("\nEnter path to image file: ").strip()
        person_name = input("Enter person's name: ").strip()
        
        if image_path and person_name:
            if self.add_famous_person_from_image(image_path, person_name):
                print(f"✅ Successfully added {person_name}!")
                print("Now when you run the live face identifier, it should recognize this person!")
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
        """Show instructions for adding real famous people."""
        print("\n📖 HOW TO ADD REAL FAMOUS PEOPLE")
        print("=" * 50)
        print("\n1. 🌐 Find photos online:")
        print("   - Search Google Images for 'Elon Musk photo'")
        print("   - Look for clear, front-facing photos")
        print("   - Download high-quality images")
        
        print("\n2. 💾 Save the images:")
        print("   - Save them in the famous_faces/ folder")
        print("   - Use descriptive names like 'elon_musk_real.jpg'")
        
        print("\n3. ➕ Add to database:")
        print("   - Use option 2 in this menu")
        print("   - Enter the image path and person's name")
        
        print("\n4. 🎯 Test recognition:")
        print("   - Run: python live_face_identifier.py")
        print("   - Show the person's photo to the camera")
        print("   - It should recognize and show their name!")
        
        print(f"\n📁 Your famous_faces folder: {self.famous_dir.absolute()}")


def main():
    """Main function."""
    print("🌟 FAMOUS FACE SETUP")
    print("=" * 40)
    
    setup = FamousFaceSetup()
    
    while True:
        print("\nOptions:")
        print("1. Setup famous faces database (create samples)")
        print("2. Add real famous person image")
        print("3. Show current database")
        print("4. Show instructions for adding real people")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            setup.setup_famous_faces()
        
        elif choice == "2":
            setup.add_real_famous_image()
        
        elif choice == "3":
            setup.show_database()
        
        elif choice == "4":
            setup.show_instructions()
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
