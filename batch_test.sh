cd /workspaces/face-detection

# Single image
python face_detection_image.py ~/Desktop/your_image.jpg ~/Desktop/output.jpg

# Batch process all images
python batch_process.py ~/Desktop/images ~/Desktop/results#!/bin/bash
# Batch process all images in a directory

INPUT_DIR="$1"
OUTPUT_DIR="${2:-.}/output"

if [ -z "$INPUT_DIR" ]; then
    echo "Usage: ./batch_test.sh <input_directory> [output_directory]"
    echo "Example: ./batch_test.sh ~/Pictures ~/Pictures/faces_detected"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Processing images from: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "---"

# Counter
count=0
detected=0

# Process all images
for image in "$INPUT_DIR"/*.{jpg,jpeg,png,JPG,JPEG,PNG}; do
    [ -e "$image" ] || continue
    
    count=$((count + 1))
    filename=$(basename "$image")
    output_file="$OUTPUT_DIR/detected_$filename"
    
    echo "$count. Processing: $filename"
    python face_detection_image.py "$image" "$output_file" 2>/dev/null
    
    if [ -f "$output_file" ]; then
        detected=$((detected + 1))
        echo "   ✓ Saved to: detected_$filename"
    fi
done

echo "---"
echo "Summary: Processed $count images, $detected results saved to $OUTPUT_DIR"
cp /workspaces/face-detection/test_result.jpg ~/Desktop/cd /workspaces/face-detection
file test_result.jpg
identify test_result.jpg  # if imagemagick installed# Download the file via the Explorer
# Or copy to a shared location

