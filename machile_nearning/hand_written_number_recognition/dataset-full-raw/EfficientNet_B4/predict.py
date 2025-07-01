# Submission ID :          274378
import modal
import os

# --- Modal Setup ---
image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1",
    "efficientnet_pytorch",
    "pillow",
    "pillow-heif"
)

app = modal.App("efficientnet-b4-predict", image=image)

VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol")
CHECKPOINT_DIR = os.path.join(VOLUME_PATH, "checkpoints_efficientnet")

# --- Prediction Function ---
@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=60*60  # 1 hour
)
def predict(data_path="/vol/evaluation_data", output_csv_path="/vol/efficientnet_b4_predictions.csv"):
    import torch
    import torch.nn as nn
    from efficientnet_pytorch import EfficientNet
    from torchvision import transforms
    from PIL import Image
    import os
    import csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Builder ---
    def build_model(num_classes, device):
        model = EfficientNet.from_pretrained("efficientnet-b4", num_classes=num_classes)
        model = model.to(device)
        return model

    # --- Preprocessing (matching training transforms) ---
    test_transforms = transforms.Compose([
        transforms.Resize((380, 380)),  # Same as training input_size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Same as training normalization
    ])

    # --- Load Model ---
    num_classes = 10
    model = build_model(num_classes, device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # --- Predict Images in data_path ---
    if not os.path.exists(data_path):
        print(f"Data path not found: {data_path}")
        return
    
    results = []
    processed_count = 0
    
    for image_file in sorted(os.listdir(data_path)):
        image_path = os.path.join(data_path, image_file)
        
        # Skip if not a file
        if not os.path.isfile(image_path):
            continue
            
        # Skip if not an image file or hidden file
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.heic', '.heif')) or image_file.startswith('.'):
            continue
            
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = test_transforms(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label = predicted.item()
            
            results.append([image_file, predicted_label])
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images...")
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

    print(f"Total images processed: {processed_count}")

    # --- Write to CSV ---
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    print(f"Predictions saved to {output_csv_path}")
    print(f"Sample predictions (first 10):")
    for i, (filename, pred) in enumerate(results[:10]):
        print(f"  {filename}: {pred}")

# --- Local Entrypoint for Modal ---
@app.local_entrypoint()
def main():
    # You can change the data_path and output_csv_path here if needed
    predict.remote("/vol/evaluation_data", "/vol/efficientnet_b4_predictions.csv")

if __name__ == "__main__":
    main()
