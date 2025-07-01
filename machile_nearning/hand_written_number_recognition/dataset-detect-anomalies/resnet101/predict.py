# Submission ID :       272723
import modal
import os

# --- Modal Setup ---
image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1",
    "pillow"
)

app = modal.App("pytorch-digit-predict", image=image)

VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol")
CHECKPOINT_DIR = os.path.join(VOLUME_PATH, "checkpoints")

# --- Prediction Function ---
@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=60*60  # 1 hour
)
def predict(data_path="/vol/evaluation_data", output_csv_path="/vol/predictions.csv"):
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image
    import os
    import csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model Builder ---
    def build_model(num_classes, device):
        model = models.resnet101(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        return model

    # --- Preprocessing ---
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- Load Model ---
    num_classes = 10
    model = build_model(num_classes, device)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval()

    # --- Predict Images in data_path ---
    results = []
    for image_file in sorted(os.listdir(data_path)):
        image_path = os.path.join(data_path, image_file)
        if not os.path.isfile(image_path):
            continue
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')) or image_file.startswith('.'):
            continue
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = test_transforms(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label = predicted.item()
            results.append([image_file, predicted_label])
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # --- Write to CSV ---
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

    print(f"Predictions saved to {output_csv_path}")

# --- Local Entrypoint for Modal ---
@app.local_entrypoint()
def main():
    # You can change the data_path and output_csv_path here if needed
    predict.remote("/vol/evaluation_data", "/vol/predictions.csv")
