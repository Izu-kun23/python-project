# Famous Face Identifier

A Python application that can identify famous people like Elon Musk, Jeff Bezos, Bill Gates, and others in images or live video streams using computer vision and face recognition.

## Features

- **Face Detection**: Automatically detects faces in images or video streams
- **Famous Person Recognition**: Identifies known celebrities and public figures
- **Real-time Processing**: Works with webcam for live identification
- **Confidence Scoring**: Provides confidence levels for each identification
- **Database Management**: Easy to add new famous people to the database
- **Multiple Face Support**: Can identify multiple people in a single image

## Quick Start

### Easy Setup (Recommended)
```bash
# Make the start script executable and run it
chmod +x start.sh
./start.sh
```

### Manual Setup
1. **Create virtual environment:**
   ```bash
   python3 -m venv face_identifier_env
   source face_identifier_env/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy Pillow imutils
   ```

3. **Run the application:**
   ```bash
   python quick_start.py
   ```

## Usage

### Quick Start Menu
Run `python quick_start.py` to access all applications:

1. **🚀 Working Face Identifier (Recommended)**
   - Main application with face detection and recognition
   - Works with webcam and images
   - Add famous people to database

2. **📸 Simple Face Detector**
   - Basic face detection for testing
   - Good for camera functionality tests

3. **🎬 Demo Application**
   - Demonstrates basic functionality
   - Perfect for first-time users

### Working Face Identifier Features:
- **Identify faces in images**: Upload an image and detect faces
- **Live webcam identification**: Real-time face detection and recognition
- **Add famous people**: Build your database of known faces
- **Database statistics**: View your face database status

## Adding Famous People

The application creates a `famous_people/` directory with subdirectories for each person:

```
famous_people/
├── elon_musk/
├── jeff_bezos/
├── bill_gates/
├── steve_jobs/
└── mark_zuckerberg/
```

To add a new famous person:
1. Create a new folder in `famous_people/` with their name (use underscores instead of spaces)
2. Add multiple clear photos of the person in the folder
3. The more photos you add, the better the recognition will be
4. Run the application and it will automatically load the new person

## Technical Details

- **Face Detection**: Uses OpenCV's Haar Cascade classifier
- **Face Recognition**: Uses the `face_recognition` library (based on dlib)
- **Database**: Stores face encodings in a pickle file for fast loading
- **Confidence Threshold**: Faces with similarity > 0.6 are considered matches

## Requirements

- Python 3.7+
- OpenCV
- face_recognition library
- NumPy
- Pillow

## Troubleshooting

### Common Issues:

1. **"No module named 'face_recognition'"**
   - Install dlib first: `pip install dlib`
   - Then install face_recognition: `pip install face_recognition`

2. **Webcam not working**
   - Make sure your webcam is connected and not being used by another application
   - Try changing the video source number (default is 0)

3. **Poor recognition accuracy**
   - Add more high-quality photos of each person
   - Ensure photos have good lighting and clear faces
   - Photos should show the person from different angles

## Example Usage

```python
from famous_face_identifier import FamousFaceIdentifier

# Create identifier instance
identifier = FamousFaceIdentifier()

# Identify faces in an image
results = identifier.identify_faces_in_image("path/to/image.jpg")
for result in results:
    print(f"Found: {result['name']} (confidence: {result['confidence']:.2f})")

# Start live webcam identification
identifier.identify_faces_in_video_stream()
```

## Privacy Note

This application processes images locally on your machine. No data is sent to external servers. The face recognition database is stored locally in your project directory.
