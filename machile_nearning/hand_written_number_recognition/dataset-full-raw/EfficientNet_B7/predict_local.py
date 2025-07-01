# Local EfficientNet-B7 Prediction Script
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os
import csv
import argparse

def build_model(num_classes, device):
    """Build EfficientNet-B7 model"""
    model = EfficientNet.from_pretrained("efficientnet-b7", num_classes=num_classes)
    model = model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description='EfficientNet-B7 Local Prediction')
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
    
    print("Starting prediction with EfficientNet-B7...")
    for image_file in sorted(os.listdir(args.data_path)):
        image_path = os.path.join(args.data_path, image_file)
        
        # Skip if not a file
        if not os.path.isfile(image_path):
            continue
            
        # Skip if not an image file or hidden file
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.heic', '.heif')) or image_file.startswith('.'):
            continue
            
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
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
                print(f"Processed {processed_count} images...")
                
            # Memory management for large model
            del image_tensor, outputs
            if processed_count % 100 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

    print(f"Total images processed: {processed_count}")

    # --- Write to CSV ---
    with open(args.output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    print(f"Predictions saved to {args.output_csv}")
    print(f"Sample predictions (first 10):")
    for i, (filename, pred) in enumerate(results[:10]):
        print(f"  {filename}: {pred}")

if __name__ == "__main__":
    main()
