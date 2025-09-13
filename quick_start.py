#!/usr/bin/env python3
"""
Quick Start Script for Face Identifier
This script sets up everything and provides easy access to all applications.
"""

import os
import sys
from pathlib import Path


def check_environment():
    """Check if we're in the virtual environment."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment is active")
        return True
    else:
        print("⚠ Virtual environment not detected")
        print("Please run: source face_identifier_env/bin/activate")
        return False


def show_menu():
    """Show the main menu."""
    print("\n" + "="*60)
    print("🎯 FACE IDENTIFIER - QUICK START MENU")
    print("="*60)
    print("\nChoose an application to run:")
    print("\n1. 🚀 Working Face Identifier (Recommended)")
    print("   - Uses OpenCV for reliable face detection")
    print("   - Works with webcam and images")
    print("   - Basic face recognition capabilities")
    
    print("\n2. 📸 Simple Face Detector")
    print("   - Basic face detection only")
    print("   - Good for testing camera functionality")
    
    print("\n3. 🎬 Demo Application")
    print("   - Demonstrates basic functionality")
    print("   - Good for first-time users")
    
    print("\n4. ⚙️  Setup & Dependencies")
    print("   - Check and install dependencies")
    print("   - Create necessary folders")
    
    print("\n5. 📊 Database Statistics")
    print("   - Show current face database status")
    
    print("\n6. 📁 Open Project Folder")
    print("   - View project files in Finder")
    
    print("\n7. ❓ Help & Documentation")
    print("   - View README and usage instructions")
    
    print("\n8. 🚪 Exit")
    
    print("\n" + "="*60)


def run_working_identifier():
    """Run the working face identifier."""
    print("\n🚀 Starting Working Face Identifier...")
    print("This is the main application with face detection and recognition!")
    os.system("python working_face_identifier.py")


def run_simple_detector():
    """Run the simple face detector."""
    print("\n📸 Starting Simple Face Detector...")
    print("This version focuses on basic face detection.")
    os.system("python simple_face_detector.py")


def run_demo():
    """Run the demo application."""
    print("\n🎬 Starting Demo Application...")
    print("This will show you the basic functionality.")
    os.system("python demo.py")


def run_setup():
    """Run the setup script."""
    print("\n⚙️ Running Setup...")
    os.system("python setup.py")


def show_database_stats():
    """Show database statistics."""
    print("\n📊 Checking Database Statistics...")
    
    # Check if working identifier exists
    if os.path.exists("working_face_identifier.py"):
        print("Running database check...")
        os.system("python -c \"from working_face_identifier import WorkingFaceIdentifier; w = WorkingFaceIdentifier(); w.show_database_stats()\"")
    else:
        print("Working face identifier not found.")


def open_project_folder():
    """Open the project folder in Finder."""
    print("\n📁 Opening project folder in Finder...")
    current_dir = os.getcwd()
    os.system(f"open '{current_dir}'")


def show_help():
    """Show help and documentation."""
    print("\n❓ HELP & DOCUMENTATION")
    print("="*40)
    
    if os.path.exists("README.md"):
        print("📖 README.md found! Here's a summary:")
        print("\nKey Features:")
        print("• Face detection in images and video streams")
        print("• Face recognition for famous people")
        print("• Webcam integration for real-time detection")
        print("• Database management for known faces")
        
        print("\nQuick Start:")
        print("1. Add photos of famous people to famous_people/ folders")
        print("2. Run the Working Face Identifier (option 1)")
        print("3. Choose option 2 for live webcam detection")
        
        print("\nSupported Image Formats:")
        print("• .jpg, .jpeg, .png")
        
        print("\nTips:")
        print("• More photos = better recognition")
        print("• Use clear, well-lit photos")
        print("• Press 'q' to quit webcam streams")
        
        print(f"\n📁 Project folder: {os.getcwd()}")
        print("📄 README.md contains detailed instructions")
    else:
        print("README.md not found in current directory")


def main():
    """Main function."""
    print("🎯 Welcome to the Face Identifier Quick Start!")
    
    # Check environment
    check_environment()
    
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                run_working_identifier()
            elif choice == "2":
                run_simple_detector()
            elif choice == "3":
                run_demo()
            elif choice == "4":
                run_setup()
            elif choice == "5":
                show_database_stats()
            elif choice == "6":
                open_project_folder()
            elif choice == "7":
                show_help()
            elif choice == "8":
                print("\n👋 Thanks for using Face Identifier!")
                print("Goodbye!")
                break
            else:
                print("\n❌ Invalid choice! Please enter a number between 1-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
