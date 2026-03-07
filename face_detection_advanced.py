"""
Advanced Face Detection with Confidence Filtering
Uses Haar Cascade with adjustable parameters for better accuracy.
"""

import cv2
import numpy as np
from pathlib import Path


class FaceDetector:
    """
    Face detection class with configurable parameters.
    """
    
    def __init__(self, classifier_path=None):
        """
        Initialize face detector.
        
        Args:
            classifier_path (str): Path to cascade classifier XML file
        """
        if classifier_path is None:
            classifier_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(classifier_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"Could not load cascade classifier from {classifier_path}")
    
    def detect_faces(self, image, scale_factor=1.05, min_neighbors=5, 
                     min_size=(30, 30), max_size=None):
        """
        Detect faces in an image.
        
        Args:
            image (np.ndarray): Input image
            scale_factor (float): Scale factor for detection
            min_neighbors (int): Minimum neighbors for detection
            min_size (tuple): Minimum face size
            max_size (tuple): Maximum face size
            
        Returns:
            list: Detected faces as (x, y, w, h) tuples
        """
        if image is None:
            return []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            maxSize=max_size
        )
        
        return faces
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2, 
                   show_info=True):
        """
        Draw detected faces on image.
        
        Args:
            image (np.ndarray): Input image
            faces (list): Detected faces
            color (tuple): Box color (BGR)
            thickness (int): Box thickness
            show_info (bool): Show face count
            
        Returns:
            np.ndarray: Image with drawn faces
        """
        image_copy = image.copy()
        
        for idx, (x, y, w, h) in enumerate(faces):
            # Draw rectangle
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
            
            # Draw circle at center
            center = (x + w // 2, y + h // 2)
            cv2.circle(image_copy, center, 5, color, -1)
            
            # Draw face number
            cv2.putText(image_copy, f'Face {idx + 1}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show face count
        if show_info:
            cv2.putText(image_copy, f'Total Faces: {len(faces)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return image_copy
    
    def process_image(self, image_path, output_path=None, **kwargs):
        """
        Process image file and detect faces.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image
            **kwargs: Additional arguments for detect_faces
        """
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        # Detect faces
        faces = self.detect_faces(image, **kwargs)
        
        # Draw faces
        result = self.draw_faces(image, faces)
        
        # Display
        cv2.imshow('Face Detection', result)
        print(f"Detected {len(faces)} face(s)")
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save output if specified
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Saved to {output_path}")
        
        return faces
    
    def process_video(self, source=0, output_path=None, **kwargs):
        """
        Process video or webcam feed.
        
        Args:
            source (int or str): 0 for webcam or path to video
            output_path (str): Path to save output video
            **kwargs: Additional arguments for detect_faces
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open source {source}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print("Processing... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect and draw faces
            faces = self.detect_faces(frame, **kwargs)
            result = self.draw_faces(frame, faces)
            
            # Display
            cv2.imshow('Face Detection', result)
            
            if writer:
                writer.write(result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames")
        if output_path:
            print(f"Saved to {output_path}")


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Advanced Face Detection")
        print("\nUsage:")
        print("  Image:  python face_detection_advanced.py image.jpg [output.jpg]")
        print("  Video:  python face_detection_advanced.py video.mp4 [output.mp4]")
        print("  Webcam: python face_detection_advanced.py webcam [output.mp4]")
        return
    
    source = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    detector = FaceDetector()
    
    if source.lower() == 'webcam':
        detector.process_video(0, output)
    elif Path(source).is_file():
        if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            detector.process_image(source, output)
        else:
            detector.process_video(source, output)
    else:
        print(f"Error: File not found {source}")


if __name__ == "__main__":
    main()
