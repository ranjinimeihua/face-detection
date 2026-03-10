# Test Script for Face Detection

## Introduction
This script demonstrates how to perform face detection on a sample image using the face detection library.

## Required Libraries
Make sure to install the required libraries before running this script:
```bash
pip install face_recognition opencv-python
```

## Sample Image
Place a sample image in the same directory as this script and name it `sample.jpg`. You can download a sample image from the internet or use any image containing a face.

## Usage
Run the following command in your terminal:
```bash
python test_face_recognition.py
```

## Script
```python
import face_recognition
import cv2
import matplotlib.pyplot as plt

# Load the sample image and learn how to recognize it.
image = face_recognition.load_image_file("sample.jpg")
face_locations = face_recognition.face_locations(image)

# Convert the image to BGR format for OpenCV to display it correctly
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Print the results
print(f"Found {len(face_locations)} face(s) in this photograph.")

# Display the results on the image
for face_location in face_locations:
    top, right, bottom, left = face_location
    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

# Show the image with the detected faces
plt.imshow(image_bgr)
plt.axis('off')  # Hide axes
plt.show()
```