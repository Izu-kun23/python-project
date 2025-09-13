#!/usr/bin/env python3
"""
Celebrity Photo Manager
Download and manage celebrity photos for professional recognition
"""

import cv2
import numpy as np
import os
import pickle
import requests
from pathlib import Path
import time
from urllib.parse import urlparse


class CelebrityPhotoManager:
    def __init__(self):
        """Initialize the celebrity photo manager."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.db_file = "professional_face_db.pkl"
        self.face_database = self.load_database()
        
        # Create directories
        self.photos_dir = Path("celebrity_photos")
        self.photos_dir.mkdir(exist_ok=True)
        
        # Celebrity photo URLs (these are example URLs - you'll need to find real ones)
        self.celebrity_photos = {
            'Elon Musk': [
                'https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg/220px-Elon_Musk_Royal_Society_%28crop2%29.jpg',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Elon_Musk_Royal_Society_%28crop1%29.jpg/220px-Elon_Musk_Royal_Society_%28crop1%29.jpg'
            ],
            'Jeff Bezos': [
                'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Jeff_Bezos_at_Amazon_Spheres_Grand_Opening_in_Seattle_-_2018_%2839074799225%29_%28cropped%29.jpg/220px-Jeff_Bezos_at_Amazon_Spheres_Grand_Opening_in_Seattle_-_2018_%2839074799225%29_%28cropped%29.jpg'
            ],
            'Bill Gates': [
                'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Bill_Gates_2018.jpg/220px-Bill_Gates_2018.jpg'
            ],
            'Davido': [
                'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Davido_2018.jpg/220px-Davido_2018.jpg'
            ],
            'Burna Boy': [
                'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Burna_Boy_2019.jpg/220px-Burna_Boy_2019.jpg'
            ],
            'Wizkid': [
                'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Wizkid_2018.jpg/220px-Wizkid_2018.jpg'
            ]
        }
        
        # Headers for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def load_database(self):
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
            'face_categories': [],
            'face_descriptions': [],
            'recognition_history': []
        }
    
    def save_database(self):
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
        features.extend(hist.flatten() / np.sum(hist))
        
        # Edge features
        edges = cv2.Canny(face_resized, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Texture features
        laplacian_var = cv2.Laplacian(face_resized, cv2.CV_64F).var()
        features.append(laplacian_var / 1000.0)
        
        return np.array(features)
    
    def download_celebrity_photo(self, name, url):
        """Download a celebrity photo from URL."""
        try:
            print(f"📥 Downloading {name} from {url}...")
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Create celebrity directory
            celebrity_dir = self.photos_dir / name.lower().replace(" ", "_")
            celebrity_dir.mkdir(exist_ok=True)
            
            # Generate filename
            parsed_url = urlparse(url)
            filename = f"{name.lower().replace(' ', '_')}_{len(list(celebrity_dir.glob('*')))}.jpg"
            filepath = celebrity_dir / filename
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded {name} as {filename}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")
            return None
    
    def process_celebrity_photo(self, image_path, name):
        """Process a celebrity photo and add to database."""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False
        
        print(f"📸 Processing {name} from {image_path}...")
        
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
        
        # Determine category
        category = self.get_celebrity_category(name)
        
        # Add to database
        self.face_database['face_features'].append(features)
        self.face_database['face_names'].append(name)
        self.face_database['face_categories'].append(category)
        self.face_database['face_descriptions'].append(f"From {os.path.basename(image_path)}")
        
        # Save database
        self.save_database()
        
        print(f"✅ Added {name} to database!")
        return True
    
    def get_celebrity_category(self, name):
        """Get category for a celebrity."""
        categories = {
            'Tech Leaders': ['Elon Musk', 'Jeff Bezos', 'Bill Gates', 'Mark Zuckerberg', 'Tim Cook'],
            'Nigerian Artists': ['Davido', 'Burna Boy', 'Wizkid', 'Tiwa Savage', 'Olamide', 'Wande Coal'],
            'Hollywood Stars': ['Leonardo DiCaprio', 'Tom Hanks', 'Scarlett Johansson', 'Ryan Reynolds'],
            'Musicians': ['Taylor Swift', 'Drake', 'Beyoncé', 'Rihanna', 'Adele'],
            'Politicians': ['Barack Obama', 'Donald Trump', 'Joe Biden'],
            'Other Celebrities': ['Oprah Winfrey', 'Ellen DeGeneres', 'Kanye West']
        }
        
        for category, celebrities in categories.items():
            if name in celebrities:
                return category
        return "Other Celebrities"
    
    def download_all_celebrity_photos(self):
        """Download all celebrity photos."""
        print("🌟 DOWNLOADING CELEBRITY PHOTOS")
        print("=" * 40)
        
        downloaded_count = 0
        
        for name, urls in self.celebrity_photos.items():
            print(f"\n📸 Processing {name}...")
            
            for url in urls:
                filepath = self.download_celebrity_photo(name, url)
                if filepath:
                    if self.process_celebrity_photo(filepath, name):
                        downloaded_count += 1
                        break  # Only use first successful photo per celebrity
        
        print(f"\n✅ Downloaded and processed {downloaded_count} celebrity photos!")
    
    def add_custom_photo(self):
        """Add a custom celebrity photo."""
        print("\n📸 ADD CUSTOM CELEBRITY PHOTO")
        print("=" * 40)
        
        print("Instructions:")
        print("1. Download a photo of a celebrity from the internet")
        print("2. Save it to your computer")
        print("3. Enter the path to the image below")
        
        image_path = input("\nEnter path to image file: ").strip()
        person_name = input("Enter celebrity's name: ").strip()
        
        if image_path and person_name:
            if self.process_celebrity_photo(image_path, person_name):
                print(f"✅ Successfully added {person_name}!")
                print("🎯 Now the professional recognition system will recognize this celebrity!")
            else:
                print(f"❌ Failed to add {person_name}")
        else:
            print("❌ Please provide both image path and person name")
    
    def show_database_status(self):
        """Show database status."""
        print("\n📊 CELEBRITY DATABASE STATUS")
        print("=" * 40)
        
        print(f"Total faces in database: {len(self.face_database['face_names'])}")
        
        if self.face_database['face_names']:
            unique_names = set(self.face_database['face_names'])
            print(f"Unique celebrities: {len(unique_names)}")
            
            # Group by category
            categories = {}
            for i, name in enumerate(self.face_database['face_names']):
                cat = self.face_database['face_categories'][i]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(name)
            
            for cat, names in categories.items():
                print(f"\n{cat}:")
                for name in sorted(set(names)):
                    count = self.face_database['face_names'].count(name)
                    print(f"  - {name}: {count} photos")
        else:
            print("Database is empty")
    
    def show_instructions(self):
        """Show instructions for adding celebrity photos."""
        print("\n📖 HOW TO ADD CELEBRITY PHOTOS")
        print("=" * 50)
        
        print("\n1. 🌐 Find photos online:")
        print("   - Go to Google Images")
        print("   - Search for 'Elon Musk photo' or 'Davido photo'")
        print("   - Look for clear, front-facing photos")
        print("   - Download high-quality images")
        
        print("\n2. 💾 Save the images:")
        print("   - Save them anywhere on your computer")
        print("   - Remember the file path")
        
        print("\n3. ➕ Add to database:")
        print("   - Use option 2 in this menu")
        print("   - Enter the image path and celebrity's name")
        
        print("\n4. 🎯 Test recognition:")
        print("   - Run: python professional_face_recognition.py")
        print("   - Show the celebrity's photo to the camera")
        print("   - It should recognize and show their name!")
        
        print("\n💡 Tips for better recognition:")
        print("   - Use clear, well-lit photos")
        print("   - Front-facing photos work best")
        print("   - Multiple photos of the same celebrity improve accuracy")
        print("   - High-resolution images work better")
        
        print(f"\n📁 Photos folder: {self.photos_dir.absolute()}")


def main():
    """Main function."""
    print("🌟 CELEBRITY PHOTO MANAGER")
    print("=" * 40)
    print("Download and manage celebrity photos for professional recognition")
    
    manager = CelebrityPhotoManager()
    
    while True:
        print("\nOptions:")
        print("1. Download celebrity photos (automatic)")
        print("2. Add custom celebrity photo")
        print("3. Show database status")
        print("4. Show instructions")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            manager.download_all_celebrity_photos()
        
        elif choice == "2":
            manager.add_custom_photo()
        
        elif choice == "3":
            manager.show_database_status()
        
        elif choice == "4":
            manager.show_instructions()
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
