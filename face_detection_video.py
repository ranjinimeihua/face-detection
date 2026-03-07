"""
Face Detection from Video or Webcam
Real-time face detection on video files or webcam feed.
"""

import cv2
import sys

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def detect_faces_in_video(source=0, output_path=None):
    """
    Detect faces in video or webcam feed.
    
    Args:
        source (int or str): 0 for webcam, or path to video file
        output_path (str): Path to save the output video (optional)
    """
    # Open video source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    face_count = 0
    
    print("Starting face detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_count += len(faces)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)
        
        # Display frame info
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Frame: {frame_count}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Detection', frame)
        
        # Write to output video if specified
        if writer:
            writer.write(frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total faces detected: {face_count}")
    if output_path:
        print(f"Output saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        detect_faces_in_video(source, output_path)
    else:
        print("Face Detection - Webcam or Video")
        print("\nUsage:")
        print("  Webcam: python face_detection_video.py")
        print("  Video:  python face_detection_video.py <video_path> [output_path]")
        print("\nExample:")
        print("  python face_detection_video.py video.mp4 output.mp4")
        print("\nPress 'q' to quit during detection")
        
        # Start with webcam by default
        detect_faces_in_video(0)
