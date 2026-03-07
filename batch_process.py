"""
Batch process images from a directory
"""

import cv2
import os
import sys
from pathlib import Path

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces_batch(input_dir, output_dir=None):
    """
    Process all images in a directory
    
    Args:
        input_dir (str): Directory containing images
        output_dir (str): Directory to save results (optional)
    """
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' not found")
        return
    
    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    
    # Find all images
    images = []
    for ext in extensions:
        images.extend(input_path.glob(ext))
    
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(images)} image(s) to process\n")
    print("=" * 60)
    
    total_faces = 0
    
    for idx, image_file in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] Processing: {image_file.name}")
        
        # Read image
        image = cv2.imread(str(image_file))
        
        if image is None:
            print(f"  ✗ Error loading image")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        print(f"  ✓ Detected {len(faces)} face(s)")
        total_faces += len(faces)
        
        # Draw rectangles around faces
        for idx_face, (x, y, w, h) in enumerate(faces, 1):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(image, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)
            cv2.putText(image, f'Face {idx_face}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save output if specified
        if output_dir:
            output_file = output_path / f"detected_{image_file.name}"
            cv2.imwrite(str(output_file), image)
            print(f"  → Saved to: {output_file.name}")
    
    print("\n" + "=" * 60)
    print(f"\nSummary:")
    print(f"  Total images processed: {len(images)}")
    print(f"  Total faces detected: {total_faces}")
    if output_dir:
        print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Batch Face Detection")
        print("\nUsage:")
        print("  python batch_process.py <input_directory> [output_directory]")
        print("\nExample:")
        print("  python batch_process.py ~/Pictures ~/Pictures/faces_detected")
        print("  python batch_process.py ./images ./output")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    detect_faces_batch(input_dir, output_dir)
