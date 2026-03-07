# Face Detection

A comprehensive face detection solution using Python and OpenCV. Includes multiple scripts for detecting faces in images, videos, and webcam feeds.

## Features

- **Image Detection**: Detect faces in static images
- **Video Detection**: Process video files with face detection
- **Webcam Detection**: Real-time face detection from webcam
- **Advanced Detection**: Configurable parameters for better accuracy
- **Output**: Draw bounding boxes and save results

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ranjinimeihua/face-detection.git
cd face-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Image Face Detection

Detect faces in a single image:

```bash
python face_detection_image.py input.jpg
```

Save output to a file:

```bash
python face_detection_image.py input.jpg output.jpg
```

### 2. Webcam Face Detection

Start real-time face detection from webcam:

```bash
python face_detection_video.py
```

### 3. Video File Face Detection

Detect faces in a video file:

```bash
python face_detection_video.py video.mp4
```

Save processed video:

```bash
python face_detection_video.py video.mp4 output.mp4
```

### 4. Advanced Face Detection

Use the advanced detector with custom parameters:

```bash
python face_detection_advanced.py image.jpg
python face_detection_advanced.py video.mp4
python face_detection_advanced.py webcam
```

## How It Works

The face detection is based on **Haar Cascade Classifiers**, a machine learning approach that uses cascade functions trained on positive and negative images. The algorithm:

1. Converts input image to grayscale
2. Applies the Haar Cascade classifier
3. Generates multiple detections at different scales
4. Applies a threshold to filter weak detections
5. Draws bounding boxes around detected faces

## Parameters

Key parameters you can adjust:

- **scaleFactor**: Image pyramid scale (1.05-1.3, lower = more thorough but slower)
- **minNeighbors**: Detection threshold (4-6, higher = fewer false positives)
- **minSize**: Minimum face size to detect (default: 30x30)
- **maxSize**: Maximum face size to detect (optional)

## Examples

### Example 1: Process webcam and save video
```bash
python face_detection_video.py output.mp4
```
Then press 'q' to quit.

### Example 2: Batch process images
```bash
for img in *.jpg; do
    python face_detection_image.py "$img" "output_$img"
done
```

### Example 3: Process video with custom parameters
Edit `face_detection_advanced.py` and adjust detection parameters in the `process_video` call.

## Output

- Detected faces are marked with green rectangles
- Face count is displayed on the image/video
- Centers of faces are marked with green dots
- Face numbers are displayed in the output

## Performance Tips

1. **Reduce resolution** for faster processing of large videos
2. **Adjust scaleFactor** - lower values are more thorough but slower
3. **Increase minNeighbors** to reduce false positives
4. **Use GPU acceleration** with CUDA-enabled OpenCV (if available)

## Limitations

- Works best with frontal faces
- Struggles with extreme angles or occlusions
- May have false positives/negatives depending on image quality
- Single-scale approach (no multi-scale detection like some advanced methods)

## Advanced Features

The `face_detection_advanced.py` script provides a class-based interface:

```python
from face_detection_advanced import FaceDetector

detector = FaceDetector()
image = cv2.imread('photo.jpg')
faces = detector.detect_faces(image, scale_factor=1.1, min_neighbors=6)
result = detector.draw_faces(image, faces)
```

## Troubleshooting

### No faces detected
- Try adjusting `scaleFactor` (use 1.05 for more detections)
- Lower `minNeighbors` value
- Ensure image quality is acceptable

### Too many false positives
- Increase `minNeighbors` (try 6-8)
- Increase `scaleFactor` (try 1.2-1.3)
- Increase `minSize` for minimum face size

### Video processing is slow
- Reduce video resolution
- Increase `scaleFactor`
- Increase `minNeighbors`
- Use GPU acceleration if available

## License

MIT License - Feel free to use this code for personal and commercial projects.

## References

- [OpenCV Face Detection](https://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection_in_an_image.html)
- [Haar Cascades](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)
- [OpenCV Documentation](https://docs.opencv.org/)

---

Created with ❤️ for face detection enthusiasts