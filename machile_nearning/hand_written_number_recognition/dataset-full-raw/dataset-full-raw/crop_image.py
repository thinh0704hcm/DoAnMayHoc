import cv2
import numpy as np
import sys
import os
from PIL import Image

BLOCK_SIZE = 50
THRESHOLD = 25

def preprocess(image):
    """Your existing preprocessing function"""
    image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return 255 - image

def postprocess(image):
    """Your existing postprocessing function"""
    image = cv2.medianBlur(image, 5)
    return image

def get_block_index(image_shape, yx, block_size): 
    """Your existing block index function"""
    y = np.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return tuple(np.meshgrid(y, x))

def adaptive_median_threshold(img_in):
    """Your existing adaptive threshold function"""
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < THRESHOLD] = 255
    return img_out

def block_image_process(image, block_size):
    """Your existing block processing function"""
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])
    return out_image

def remove_grid_lines(binary_img):
    """Remove horizontal and vertical grid lines using morphological operations"""
    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    gridless = cv2.subtract(binary_img, detected_lines)
    
    # Remove vertical lines  
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(gridless, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    gridless = cv2.subtract(gridless, detected_lines)
    
    return gridless

def extract_digits_from_image(binary_img, min_width=10, min_height=15):
    """Extract individual digits using contour detection"""
    # Find contours[7]
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_regions = []
    bounding_boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter contours by size to get likely digits[7]
        if h > min_height and w > min_width and w < h * 2:  # Aspect ratio check
            # Extract the digit region
            roi = binary_img[y:y+h, x:x+w]
            
            # Add padding around the digit
            padded_roi = cv2.copyMakeBorder(roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
            
            # Resize to standard size (28x28 for MNIST compatibility)[5]
            resized = cv2.resize(padded_roi, (28, 28))
            
            digit_regions.append(resized)
            bounding_boxes.append((x, y, w, h))
    
    # Sort digits left to right, top to bottom[7]
    if bounding_boxes:
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        sorted_indices = sorted(range(len(bounding_boxes)), 
                              key=lambda i: (bounding_boxes[i][1] // 30, bounding_boxes[i][0]))
        
        sorted_digits = [digit_regions[i] for i in sorted_indices]
        sorted_boxes = [bounding_boxes[i] for i in sorted_indices]
        
        return sorted_digits, sorted_boxes
    
    return [], []

def process_image_with_digit_extraction(filename):
    """Main processing function combining your pipeline with digit extraction"""
    # Load image
    image_in = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
    
    # Your existing preprocessing pipeline
    preprocessed = preprocess(image_in)
    thresholded = block_image_process(preprocessed, BLOCK_SIZE)
    postprocessed = postprocess(thresholded)
    
    # NEW STEPS: Grid removal and digit extraction
    # Remove grid lines
    gridless = remove_grid_lines(postprocessed)
    
    # Extract digits
    digit_images, digit_boxes = extract_digits_from_image(gridless)
    
    # Save results
    base_name = os.path.splitext(filename)[0]
    
    # Save intermediate results for debugging
    cv2.imwrite(f'{base_name}_1_thresholded.png', postprocessed)
    cv2.imwrite(f'{base_name}_2_gridless.png', gridless)
    
    # Save individual digits
    os.makedirs(f'{base_name}_digits', exist_ok=True)
    
    for i, (digit_img, box) in enumerate(zip(digit_images, digit_boxes)):
        digit_filename = f'{base_name}_digits/digit_{i:03d}.png'
        cv2.imwrite(digit_filename, digit_img)
        print(f"Extracted digit {i}: {digit_filename} (bbox: {box})")
    
    print(f"Total digits extracted: {len(digit_images)}")
    
    return digit_images, digit_boxes

def visualize_detection(filename):
    """Create visualization showing detected digit locations"""
    # Load original image
    original = cv2.imread(filename)
    
    # Process to get digit locations
    digit_images, digit_boxes = process_image_with_digit_extraction(filename)
    
    # Draw bounding boxes on original image
    for i, (x, y, w, h) in enumerate(digit_boxes):
        cv2.rectangle(original, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 2)
        cv2.putText(original, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    base_name = os.path.splitext(filename)[0]
    cv2.imwrite(f'{base_name}_detection.png', original)
    print(f"Detection visualization saved: {base_name}_detection.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Process image and extract digits
    digit_images, digit_boxes = process_image_with_digit_extraction(filename)
    
    # Create detection visualization
    visualize_detection(filename)
