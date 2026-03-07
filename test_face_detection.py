find ~ -name "images" -type d 2>/dev/null | head -5"""
Test script for face detection on a headless environment
"""

import cv2
import os

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def test_face_detection(image_path, output_path='output.jpg'):
    """Test face detection on an image"""
    
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    print(f"Image shape: {image.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale")
    
    # Detect faces
    print("Detecting faces...")
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    print(f"✓ Detected {len(faces)} face(s)")
    
    # Draw rectangles around faces
    for idx, (x, y, w, h) in enumerate(faces, 1):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(image, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)
        cv2.putText(image, f'Face {idx}', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"  Face {idx}: Position (x={x}, y={y}), Size {w}x{h}")
    
    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"✓ Output saved to {output_path}")
    
    return True

if __name__ == "__main__":
    # Test on the downloaded image
    test_face_detection('test_image.jpg', 'test_output.jpg')
    
    # Check output file
    if os.path.exists('test_output.jpg'):
        size = os.path.getsize('test_output.jpg')
        print(f"\n✓ Test successful! Output file size: {size} bytes")
    else:
        print("\n✗ Test failed! Output file not created")
find ~ -name "images" -type d 2>/dev/null | head -5python3 << 'EOF'
from pathlib import Path
import os

# Check where files might be
home = Path.home()
print(f"Home directory: {home}\n")
print("Looking for desktop or images folder...")

# Search for Desktop or images folder
for root, dirs, files in os.walk(home):
    if 'Desktop' in dirs or 'images' in dirs:
        desktop_path = Path(root)
        print(f"Found: {desktop_path}")
        
        # List subdirectories
        for subdir in dirs[:5]:
            print(f"  - {subdir}/")
        break
EOF
# Copy all images from desktop images folder
cp ~/Desktop/images/*.jpg /workspaces/face-detection/desktop_images/
cp ~/Desktop/images/*.png /workspaces/face-detection/desktop_images/ 2>/dev/null

# Then run batch process
python batch_process.py /workspaces/face-detection/desktop_images /workspaces/face-detection/desktop_resultscd /workspaces/face-detection
python test_face_detection.py
ls -lh test_output.jpg
ls -lh test_output.jpg

