# Submission ID :    272804
import modal
import os

# --- Modal Setup ---
image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1",
    "efficientnet_pytorch",
    "timm",
    "pillow"
)

app = modal.App("consensus-predict", image=image)
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol")

# --- Model Configuration ---
MODEL_PATHS = {
    'resnet50': '/vol/checkpoints_resnet50/latest.pth',
    'resnet101': '/vol/checkpoints/latest.pt',
    'efficientnet_b4': '/vol/checkpoints_efficientnet/latest.pt',
    'vit_base': '/vol/checkpoints_vit/latest.pt'
}

# --- Prediction Function ---
@app.function(
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=60*60
)
def consensus_predict(data_path="/vol/evaluation_data", output_csv_path="/vol/consensus_predictions.csv"):
    import torch
    import numpy as np
    from torchvision import transforms, models
    from efficientnet_pytorch import EfficientNet
    import timm
    from PIL import Image
    import csv

    device = torch.device("cuda")
    
    # --- Transforms ---
    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    effnet_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- Model Loading ---
    def load_model(model_name):
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
        elif model_name == 'efficientnet_b4':
            model = EfficientNet.from_name('efficientnet-b4', num_classes=10)
        elif model_name == 'vit_base':
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
        
        if model_name == 'resnet50':
            model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location=device))
        else:
            model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location=device)['model_state_dict'])
        
        model.eval()
        return model.to(device)

    # Load all models
    models_dict = {
        name: load_model(name) for name in MODEL_PATHS.keys()
    }

    # --- Prediction Loop ---
    results = []
    for image_file in sorted(os.listdir(data_path)):
        image_path = os.path.join(data_path, image_file)
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            img = Image.open(image_path).convert('RGB')
            
            # Prepare inputs for different models
            resnet_input = resnet_transform(img).unsqueeze(0).to(device)
            effnet_input = effnet_transform(img).unsqueeze(0).to(device)
            
            # Get predictions from all models
            with torch.no_grad():
                preds = {
                    'resnet50': models_dict['resnet50'](resnet_input).softmax(1),
                    'resnet101': models_dict['resnet101'](resnet_input).softmax(1),
                    'efficientnet_b4': models_dict['efficientnet_b4'](effnet_input).softmax(1),
                    'vit_base': models_dict['vit_base'](resnet_input).softmax(1)
                }

            # Collect predictions and confidences
            model_preds = [p.argmax().item() for p in preds.values()]
            confidences = [p.max().item() for p in preds.values()]
            
            # Calculate consensus
            counts = np.bincount(model_preds, minlength=10)
            consensus = np.argmax(counts)
            consensus_strength = np.max(counts) / len(model_preds)

            results.append([
                image_file,
                consensus,
            ])

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # --- Write Results ---
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

    print(f"Consensus predictions saved to {output_csv_path}")

# --- Local Entrypoint ---
@app.local_entrypoint()
def main():
    consensus_predict.remote()
