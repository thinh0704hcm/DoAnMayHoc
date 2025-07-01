# Enhanced HEIC-compatible EfficientNet-B7 Local Prediction Script
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os
import csv
import argparse

# Try to import HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
    print("✓ HEIC/HEIF support enabled")
except ImportError:
    HEIC_SUPPORT = False
    print("⚠ HEIC/HEIF support not available. Install with: pip install pillow-heif")

def load_image_with_heic_support(image_path):
    """Load image with HEIC support and fallback handling"""
    try:
        # Try to load with PIL (supports HEIC if pillow-heif is installed)
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        # If HEIC loading fails, try alternative methods
        if image_path.lower().endswith(('.heic', '.heif')):
            if not HEIC_SUPPORT:
                raise ImportError(f"HEIC file detected but pillow-heif not installed: {image_path}")
            else:
                raise Exception(f"Failed to load HEIC file: {image_path}. Error: {e}")
        else:
            raise Exception(f"Failed to load image: {image_path}. Error: {e}")

def build_model(num_classes, device):
    """Build EfficientNet-B7 model"""
    model = EfficientNet.from_pretrained("efficientnet-b7", num_classes=num_classes)
    model = model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description='EfficientNet-B7 Local Prediction with HEIC Support')
    parser.add_argument('--data_path', type=str, required=True, help='Path to evaluation data')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_csv', type=str, default='efficientnet_b7_predictions.csv', help='Output CSV file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for prediction (default: 1)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Preprocessing (matching training transforms) ---
    test_transforms = transforms.Compose([
        transforms.Resize((600, 600)),  # Same as training input_size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # --- Load Model ---
    num_classes = 10
    model = build_model(num_classes, device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found at {args.checkpoint_path}")
        return
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {args.checkpoint_path}")
    
    if 'best_acc' in checkpoint:
        print(f"Best accuracy from training: {checkpoint['best_acc']:.4f}")

    # --- Predict Images in data_path ---
    if not os.path.exists(args.data_path):
        print(f"Data path not found: {args.data_path}")
        return
    
    results = []
    processed_count = 0
    heic_count = 0
    error_count = 0
    
    print("Starting prediction with EfficientNet-B7...")
    
    # Supported file extensions
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    if HEIC_SUPPORT:
        supported_extensions += ('.heic', '.heif')
    
    print(f"Supported formats: {', '.join(supported_extensions)}")
    
    for image_file in sorted(os.listdir(args.data_path)):
        image_path = os.path.join(args.data_path, image_file)
        
        # Skip if not a file
        if not os.path.isfile(image_path):
            continue
            
        # Skip if not an image file or hidden file
        if not image_file.lower().endswith(supported_extensions) or image_file.startswith('.'):
            continue
        
        # Count HEIC files
        if image_file.lower().endswith(('.heic', '.heif')):
            heic_count += 1
            
        try:
            # Load and preprocess image with HEIC support
            image = load_image_with_heic_support(image_path)
            image_tensor = test_transforms(image).unsqueeze(0).to(device, non_blocking=True)
            
            # Make prediction with mixed precision
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(image_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label = predicted.item()
            
            results.append([image_file, predicted_label])
            processed_count += 1
            
            if processed_count % 50 == 0:  # Less frequent logging for B7
                print(f"Processed {processed_count} images... (HEIC: {heic_count}, Errors: {error_count})")
                
            # Memory management for large model
            del image_tensor, outputs
            if processed_count % 100 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            error_count += 1
            continue

    print(f"\n=== PREDICTION SUMMARY ===")
    print(f"Total images processed: {processed_count}")
    print(f"HEIC/HEIF images: {heic_count}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {(processed_count/(processed_count+error_count)*100):.1f}%")

    # --- Write to CSV ---
    with open(args.output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'prediction'])  # Add header
        writer.writerows(results)

    print(f"\nPredictions saved to {args.output_csv}")
    print(f"Sample predictions (first 10):")
    for i, (filename, pred) in enumerate(results[:10]):
        print(f"  {filename}: {pred}")

if __name__ == "__main__":
    main()
