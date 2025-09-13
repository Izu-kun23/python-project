#!/usr/bin/env python3
"""
Add Real Photos of Famous People
This script helps you add real photos to improve face recognition.
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path


def add_real_photo():
    """Add a real photo of a famous person."""
    print("📸 ADD REAL PHOTO OF FAMOUS PERSON")
    print("=" * 40)
    
    # Load database
    db_file = "face_database.pkl"
    if os.path.exists(db_file):
        with open(db_file, 'rb') as f:
            face_database = pickle.load(f)
    else:
        print("❌ No face database found!")
        return
    
    # Face detection setup
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def extract_face_features(face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(gray, (100, 100))
        
        features = []
        hist = cv2.calcHist([face_resized], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
        
        edges = cv2.Canny(face_resized, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(face_resized)
        features.append(len(eyes))
        
        return np.array(features)
    
    print("📋 Instructions:")
    print("1. Download a photo of a famous person from the internet")
    print("2. Save it to your computer")
    print("3. Enter the path to the image below")
    
    print(f"\n💡 Famous people you can add:")
    unique_names = set(face_database['face_names'])
    for name in sorted(unique_names):
        print(f"   - {name}")
    
    image_path = input("\nEnter path to image file: ").strip()
    person_name = input("Enter person's name: ").strip()
    
    if not image_path or not person_name:
        print("❌ Please provide both image path and person name")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📸 Processing {person_name} from {image_path}...")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        print(f"❌ No faces found in {image_path}")
        print("💡 Make sure the image contains a clear face")
        return
    
    print(f"✅ Found {len(faces)} face(s)")
    
    # Use the first face
    x, y, w, h = faces[0]
    face_roi = img[y:y+h, x:x+w]
    
    # Extract features
    features = extract_face_features(face_roi)
    
    # Add to database
    face_database['face_features'].append(features)
    face_database['face_names'].append(person_name)
    face_database['face_descriptions'].append(f"From {os.path.basename(image_path)}")
    
    # Save database
    with open(db_file, 'wb') as f:
        pickle.dump(face_database, f)
    
    print(f"✅ Successfully added {person_name} to database!")
    print("🎯 Now when you run the live face identifier, it should recognize this person!")
    
    # Show updated database
    print(f"\n📊 Updated database:")
    print(f"Total faces: {len(face_database['face_names'])}")
    unique_names = set(face_database['face_names'])
    print(f"Unique people: {len(unique_names)}")


def show_quick_guide():
    """Show quick guide for adding photos."""
    print("\n📖 QUICK GUIDE: ADDING FAMOUS PEOPLE PHOTOS")
    print("=" * 50)
    
    print("\n1. 🌐 Find photos online:")
    print("   - Go to Google Images")
    print("   - Search for 'Elon Musk photo'")
    print("   - Look for clear, front-facing photos")
    print("   - Download high-quality images")
    
    print("\n2. 💾 Save the images:")
    print("   - Save them anywhere on your computer")
    print("   - Remember the file path")
    
    print("\n3. ➕ Add to database:")
    print("   - Run this script")
    print("   - Enter the image path and person's name")
    
    print("\n4. 🎯 Test recognition:")
    print("   - Run: python live_face_identifier.py")
    print("   - Show the person's photo to the camera")
    print("   - It should recognize and show their name!")
    
    print("\n💡 Tips:")
    print("   - Use clear, well-lit photos")
    print("   - Front-facing photos work best")
    print("   - Multiple photos of the same person improve accuracy")


def main():
    """Main function."""
    print("🌟 ADD REAL PHOTOS OF FAMOUS PEOPLE")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Add real photo of famous person")
        print("2. Show quick guide")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            add_real_photo()
        
        elif choice == "2":
            show_quick_guide()
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
