import cv2
import numpy as np
import itertools
import os
import sys

from core.filter import GuidedFilter
from tools import visualize as vis
from cv.image import to_8U, to_32F

# Fixed parameters
RADIUS = 8
EPSILON = 0.32

def process_image(input_path, output_path):
    """Apply guided filter to an image and save as PNG."""
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not read {input_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for filtering
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply guided filter
    GF = GuidedFilter(gray, radius=RADIUS, eps=EPSILON)
    filtered = GF.filter(gray)
    
    # Convert back to BGR and save as PNG
    output_file = os.path.splitext(os.path.basename(input_path))[0] + ".png"
    output_full_path = os.path.join(output_path, output_file)
    cv2.imwrite(output_full_path, cv2.cvtColor(to_8U(filtered), cv2.COLOR_GRAY2BGR))
    print(f"Processed: {input_path} -> {output_full_path}")

def process_directory(input_dir, output_dir):
    """Process all image files in the input directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # Process only image files
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            process_image(input_path, output_dir)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 GIF.py input/directory output/directory")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    process_directory(input_dir, output_dir)
