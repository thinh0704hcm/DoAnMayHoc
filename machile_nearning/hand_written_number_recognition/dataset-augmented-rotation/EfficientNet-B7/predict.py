# Submission ID:        273430
import modal
import os

# --- Modal Setup ---
image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1",
    "efficientnet_pytorch",
    "pillow"
)

app = modal.App("efficientnet-b7-predict", image=image)
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol")

# --- Model Configuration ---
MODEL_PATH = '/vol/checkpoints_efficientnet_b7/best_model.pth'

# --- Prediction Function ---
@app.function(
    gpu="T4",  # Match training GPU spec
    volumes={VOLUME_PATH: volume},
    timeout=60*60
)
def predict(data_path="/vol/evaluation_data", output_csv_path="/vol/efficientnet_b7_predictions.csv"):
    import torch
    from torchvision import transforms
    from efficientnet_pytorch import EfficientNet
    from PIL import Image
    import csv
    import numpy as np

    device = torch.device("cuda")
    
    # --- Transform ---
    transform = transforms.Compose([
        transforms.Resize((600, 600)),  # Match training input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Model Loading ---
    model = EfficientNet.from_pretrained("efficientnet-b7", num_classes=10)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Handle DataParallel wrapping
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()

    # --- Prediction Loop ---
    results = []
    for image_file in sorted(os.listdir(data_path)):
        image_path = os.path.join(data_path, image_file)
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            img = Image.open(image_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()
            
            results.append([image_file, prediction, confidence])
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results.append([image_file, -1, 0.0])  # Error placeholder

    # --- Write Results ---
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'prediction', 'confidence'])
        writer.writerows(results)

    print(f"Predictions saved to {output_csv_path}")

# --- Local Entrypoint ---
@app.local_entrypoint()
def main():
    predict.remote()