#!/usr/bin/env python3
"""
Setup script for Famous Face Identifier
This script helps with initial setup and dependency installation.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'cv2',
        'numpy',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'PIL':
                from PIL import Image
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    # Check face_recognition separately with better error handling
    try:
        import face_recognition
        print("✓ face_recognition is installed")
    except ImportError as e:
        missing_packages.append('face_recognition')
        print(f"✗ face_recognition is missing: {e}")
    except Exception as e:
        print(f"⚠ face_recognition has issues: {e}")
        print("  This might still work for basic functionality")
    
    return missing_packages


def create_directories():
    """Create necessary directories."""
    directories = [
        "famous_people",
        "famous_people/elon_musk",
        "famous_people/jeff_bezos", 
        "famous_people/bill_gates",
        "famous_people/steve_jobs",
        "famous_people/mark_zuckerberg"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def main():
    """Main setup function."""
    print("=== Famous Face Identifier Setup ===\n")
    
    # Check current dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        install_choice = input("Would you like to install missing packages? (y/n): ").lower()
        
        if install_choice == 'y':
            if not install_requirements():
                print("Setup failed. Please install packages manually.")
                return
        else:
            print("Please install missing packages manually before running the application.")
            return
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Create sample README files
    sample_people = {
        "elon_musk": "Elon Musk",
        "jeff_bezos": "Jeff Bezos",
        "bill_gates": "Bill Gates", 
        "steve_jobs": "Steve Jobs",
        "mark_zuckerberg": "Mark Zuckerberg"
    }
    
    print("\nCreating sample README files...")
    for folder_name, display_name in sample_people.items():
        readme_path = Path("famous_people") / folder_name / "README.txt"
        if not readme_path.exists():
            with open(readme_path, 'w') as f:
                f.write(f"Add images of {display_name} in this folder.\n")
                f.write("Supported formats: .jpg, .jpeg, .png\n")
                f.write("The more images you add, the better the recognition will be.\n")
            print(f"✓ Created README for {display_name}")
    
    print("\n=== Setup Complete! ===")
    print("\nNext steps:")
    print("1. Add photos of famous people to the folders in 'famous_people/'")
    print("2. Run: python famous_face_identifier.py")
    print("3. Choose option 2 to test with your webcam")
    
    print("\nTip: You can find celebrity photos from:")
    print("- Official social media accounts")
    print("- News articles and interviews") 
    print("- Wikipedia (public domain images)")
    print("- Getty Images or similar (with proper licensing)")


if __name__ == "__main__":
    main()
