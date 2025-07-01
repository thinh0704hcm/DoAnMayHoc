import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Block processing parameters
THRESHOLD = 25  # For adaptive median threshold

def sharpen_image(image):
    gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=1)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened

def preprocess_block(image):
    """Initial preprocessing for block processing"""
    image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    return 255 - image  # Invert image

def postprocess_block(image):
    """Final processing after block processing"""
    return cv2.medianBlur(image, 5)

def get_block_index(image_shape, yx, block_size): 
    """Calculate block indices for adaptive processing"""
    y = np.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return tuple(np.meshgrid(y, x))

def adaptive_median_threshold(img_in):
    """Apply adaptive thresholding based on local median"""
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < THRESHOLD] = 255
    return img_out

def block_image_process(image, block_size):
    """Process image in blocks using adaptive thresholding"""
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])
    return out_image

def remove_lines_hough(binary_image, remove_diagonals=False):
    """Remove straight lines using Hough Transform with proper preprocessing."""
    # Step 1: Apply Canny edge detection
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    # Step 2: Dilate edges to connect broken lines
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Step 3: Detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=25,
        minLineLength=int(min(binary_image.shape[1]*0.1, binary_image.shape[0]*0.1)),
        maxLineGap=int(min(binary_image.shape[1]*0.01, binary_image.shape[0]*0.01))
    )

    # Prepare mask
    mask = np.zeros_like(binary_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            if remove_diagonals or abs(angle) < 15 or abs(angle) > 165 or (75 < abs(angle) < 105):
                thickness = int(binary_image.shape[1]/45)
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    # Apply mask to remove lines
    result = binary_image.copy()
    result[mask == 255] = 255  # Set line regions to white
    # Ensure result is binary
    result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)[1]
    return result


def preprocess_digit_image(image_path, debug=False):
    """Xử lý ảnh số viết tay để tách nền và loại bỏ đường kẻ"""
    # 1. Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # 2. Làm sắc nét ảnh gốc
    small_img = sharpen_image(img)
    
    # 3. Chuyển đổi ảnh sang không gian màu LAB và lấy kênh L (luminance)
    lab = cv2.cvtColor(small_img, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    
    # 4. Xoá bóng bằng cách sử dụng Gaussian Blur
    length = l_channel.shape[0]
    width = l_channel.shape[1]
    kernel_size = max(3, int(min(width,length) / 6))  # 1/6 chiều dài hoặc chiều rộng hoặc tối thiểu 3px
    if kernel_size % 2 == 0:  # Đảm bảo kích thước kernel là số lẻ
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)
    normalized = cv2.divide(l_channel, blurred, scale=255)
    
    # 5. Tính toán kích thước khối cho xử lý khối
    block_size_val = max(10, int(min(width,length) / 10))  # Tối thiểu 10px hoặc 1/10 chiều dài hoặc chiều rộng
    if block_size_val % 2 == 0:  # Đảm bảo kích thước khối là số lẻ
        block_size_val += 1
    
    # 6. Tiền xử lý ảnh cho xử lý khối
    preprocessed = preprocess_block(normalized)
    block_processed = block_image_process(preprocessed, block_size_val)
    final_thresh = postprocess_block(block_processed)

    # 7. Xử lý loại bỏ đường kẻ bằng Hough Transform
    # Lần đầu tiên
    final_no_lines = remove_lines_hough(final_thresh)
    # Lần thứ hai để loại bỏ các đường kẻ còn sót lại
    final_result = remove_lines_hough(final_no_lines)

    if debug:
        return {
            'original': small_img,
            'l_channel': l_channel,
            'normalized': normalized,
            'preprocessed': preprocessed,
            'block_processed': block_processed,
            'final_thresholded': final_thresh,
            'final_result': final_result,
            'block_size': block_size_val
        }
    
    return final_result

# Image path
# IMG_PATH = "D:/machile_nearning/hand_written_number_recognition/dataset-full-raw/dataset-full-raw/combined_data/1_b92be5ec-2b20-4e2a-9e67-f678652d4166.jpg"
# IMG_PATH = "D:/machile_nearning/hand_written_number_recognition/dataset-full-raw/dataset-full-raw/combined_data/1_b945935d-8d15-44bb-a9d7-10d6e04e6a51.jpg"
IMG_PATH = "D:/machile_nearning/hand_written_number_recognition/dataset-full-raw/dataset-full-raw/combined_data/0_47ccd998-3e82-4d26-9bfb-235c78a7b8a3.png"
# Process image in debug mode
debug_results = preprocess_digit_image(IMG_PATH, debug=True)

# Create visualization
plt.figure(figsize=(15, 10))

# Original image
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(debug_results['original'], cv2.COLOR_BGR2RGB))
plt.title('Ảnh gốc')
plt.axis('off')

# L-channel (luminance)
plt.subplot(2, 4, 2)
plt.imshow(debug_results['l_channel'], cmap='gray')
plt.title('Ảnh trắng đen (L-channel)')
plt.axis('off')

# Normalized image
plt.subplot(2, 4, 3)
plt.imshow(debug_results['normalized'], cmap='gray')
plt.title('Đồng hóa độ sáng')
plt.axis('off')

# Preprocessed for block processing
plt.subplot(2, 4, 4)
plt.imshow(debug_results['preprocessed'], cmap='gray')
plt.title('Tiền xử lý cho xử lý khối')
plt.axis('off')

# Block processed image
plt.subplot(2, 4, 5)
plt.imshow(debug_results['block_processed'], cmap='gray')
plt.title(f'Đã xử lí khối (Kích thước : {debug_results["block_size"]})')
plt.axis('off')

# Final thresholded result
plt.subplot(2, 4, 6)
plt.imshow(debug_results['final_thresholded'], cmap='gray')
plt.title('Kết quả tách nền')
plt.axis('off')

# Final without lines
plt.subplot(2, 4, 7)
plt.imshow(debug_results['final_result'], cmap='gray')
plt.title('Kết quả xóa đường kẻ')
plt.axis('off')

plt.tight_layout()
plt.show()