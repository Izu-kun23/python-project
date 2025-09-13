#!/usr/bin/env python3
"""
Webcam Demo - Simple face detection with camera permission handling
"""

import cv2
import numpy as np
import sys


def test_camera_permissions():
    """Test camera access and provide helpful messages."""
    print("🎥 Testing camera access...")
    
    # Try different camera sources
    for camera_id in [0, 1, 2]:
        print(f"Trying camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera {camera_id} is working!")
                cap.release()
                return camera_id
            else:
                print(f"❌ Camera {camera_id} opened but can't read frames")
        else:
            print(f"❌ Camera {camera_id} failed to open")
        
        cap.release()
    
    return None


def run_webcam_demo():
    """Run the webcam face detection demo."""
    print("\n🎯 WEBCAM FACE DETECTION DEMO")
    print("=" * 40)
    
    # Test camera
    camera_id = test_camera_permissions()
    
    if camera_id is None:
        print("\n❌ No working camera found!")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure your camera is connected")
        print("2. Check if another app is using the camera")
        print("3. On macOS: Go to System Preferences > Security & Privacy > Camera")
        print("4. Allow your terminal app to access the camera")
        print("5. Try restarting your terminal")
        return
    
    print(f"\n✅ Using camera {camera_id}")
    print("\n📋 Instructions:")
    print("- Your webcam will open in a new window")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    print("- Move your face in front of the camera")
    
    input("\nPress Enter to start webcam...")
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("\n🎥 Webcam started! Look at the camera window...")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read from camera")
                break
            
            frame_count += 1
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Face #{len(faces)}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add info text
            cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Face Detection Demo', frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting webcam demo...")
                break
            elif key == ord('s'):
                filename = f"webcam_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Saved frame as {filename}")
    
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam demo finished!")


def main():
    """Main function."""
    print("🎯 Welcome to the Webcam Face Detection Demo!")
    print("This will test your camera and show face detection in real-time.")
    
    run_webcam_demo()


if __name__ == "__main__":
    main()
