import os
import glob
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import pillow_heif

# Processing parameters
THRESHOLD = 25
MAX_WORKERS = os.cpu_count() or 8  # Dynamic worker count
MIN_IMAGE_SIZE = 28  # Skip processing for smaller images

# ----------------------------
# HEIC Handling (Optimized)
# ----------------------------
def load_image(img_path):
    """Load image efficiently with HEIC support"""
    ext = os.path.splitext(img_path)[1].lower()
    img = None
    
    if ext == '.heic':
        pillow_heif.register_heif_opener()
        try:
            pil_img = Image.open(img_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            # Save converted JPG only if needed
            jpg_path = os.path.splitext(img_path)[0] + ".jpg"
            if not os.path.exists(jpg_path):
                pil_img.save(jpg_path, "JPEG", quality=95)
            return img, jpg_path
        except Exception as e:
            print(f"HEIC conversion error: {img_path} - {str(e)}")
            return None, None
    else:
        img = cv2.imread(img_path)
        return img, img_path

# ----------------------------
# Image Processing (Optimized)
# ----------------------------
def sharpen_image(image):
    """Efficient sharpening with reduced operations"""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(image, 1.7, blurred, -0.7, 0)

def adaptive_median_threshold(block):
    """Vectorized adaptive thresholding"""
    med = np.median(block)
    return 255 * (block - med < THRESHOLD)

def block_image_process(image, block_size):
    """Optimized block processing with vectorization"""
    h, w = image.shape
    out_image = np.zeros_like(image)
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            block = image[y:y_end, x:x_end]
            out_image[y:y_end, x:x_end] = adaptive_median_threshold(block)
            
    return out_image

def remove_lines_hough(binary_image):
    """Efficient line removal with parameter optimization"""
    # Skip for small images
    h, w = binary_image.shape[:2]
    if h < 50 or w < 50:
        return binary_image
    
    # Optimized edge detection
    edges = cv2.Canny(binary_image, 40, 120)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Dynamic parameters based on image size
    min_line_length = max(20, int(w * 0.3))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=min_line_length,
        maxLineGap=int(w * 0.02)
    )
    
    if lines is None:
        return binary_image
    
    result = binary_image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
        # Filter lines within tolerance of horizontal/vertical
        if angle < 15 or angle > 165 or (75 < angle < 105):
            thickness = max(1, int(w / 150))
            cv2.line(result, (x1, y1), (x2, y2), 255, thickness)
            
    return result

def preprocess_digit_image(img):
    """Optimized processing pipeline"""
    h, w = img.shape[:2]
    
    # Downsample large images
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        h, w = img.shape[:2]
    
    # Efficient shadow removal
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    kernel_size = max(3, int(w / 6))
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blurred = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)
    normalized = cv2.divide(l_channel, blurred, scale=255)
    
    # Preprocessing
    preprocessed = 255 - cv2.medianBlur(normalized, 3)
    
    # Block processing
    block_size = max(10, int(w / 10))
    block_processed = block_image_process(preprocessed, block_size)
    
    # Line removal
    result = remove_lines_hough(block_processed)
    return cv2.medianBlur(result, 3)

# ----------------------------
# Parallel Execution (Optimized)
# ----------------------------
def process_single_image(args):
    """Process single image with error handling"""
    img_path, dst_folder, quality_threshold = args
    try:
        img, processed_path = load_image(img_path)
        if img is None:
            return f"Failed to load: {img_path}"
            
        # Skip small images
        if img.shape[0] <= MIN_IMAGE_SIZE or img.shape[1] <= MIN_IMAGE_SIZE:
            return f"Skipped small image: {img_path} ({img.shape[1]}x{img.shape[0]})"
        
        digit_img = preprocess_digit_image(img)
        
        # Quality check
        if np.mean(digit_img) < quality_threshold:
            return f"Low quality skipped: {img_path}"
        
        # Save result
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(dst_folder, f"{base_name}.png")
        cv2.imwrite(out_path, digit_img)
        return f"Processed: {img_path} -> {out_path}"
        
    except Exception as e:
        return f"Error processing {img_path}: {str(e)}"

def extract_digits_from_folder(src_folder, dst_folder, quality_threshold=30):
    os.makedirs(dst_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(src_folder, "*.*"))
    
    # Process images in parallel
    tasks = [(path, dst_folder, quality_threshold) for path in image_paths]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            if result:
                print(result)

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    src_folder = "D:/machile_nearning/hand_written_number_recognition/dataset-full-raw/dataset-full-raw/combined_data"
    dst_folder = "./dataset-detect-anomalies"
    extract_digits_from_folder(src_folder, dst_folder)
