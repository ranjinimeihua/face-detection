"""
Face Detection from Static Images
Detects faces in images and draws bounding boxes around them.
"""

import cv2
import sys
from pathlib import Path

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def detect_faces_in_image(image_path, output_path=None):
    """
    Detect faces in an image and optionally save the result.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the output image (optional)
    """
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert to grayscale (required for face detection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    print(f"Detected {len(faces)} face(s)")
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw circle at center
        cv2.circle(image, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)
    
    # Display the result
    cv2.imshow('Face Detection', image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the output image if specified
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Output saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_detection_image.py <image_path> [output_path]")
        print("Example: python face_detection_image.py input.jpg output.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    detect_faces_in_image(image_path, output_path)
image.jpeg
