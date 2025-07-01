import modal
import os
import csv

# Reuse existing Modal setup
image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1",
    "pillow"
)

app = modal.App("label-accuracy-check", image=image)
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol")

@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=60*60
)
def check_accuracy(confidence_threshold=0.9):
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision import transforms
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    
    # Reuse dataset class from training script
    class DigitDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_files = []
            self.labels = []
            for label_folder in os.listdir(root_dir):
                label_path = os.path.join(root_dir, label_folder)
                if os.path.isdir(label_path):
                    try:
                        label = int(label_folder)
                        for image_file in os.listdir(label_path):
                            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')) and not image_file.startswith('.'):
                                self.image_files.append(os.path.join(label_path, image_file))
                                self.labels.append(label)
                    except ValueError:
                        print(f"Skipping non-integer folder: {label_folder}")

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
    
    # Reuse model architecture from training script
    def build_model(num_classes, device):
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)
    
    # Setup transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(10, device)
    checkpoint_path = os.path.join(VOLUME_PATH, "saved_checkpoints", "latest.pt")
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset = DigitDataset(
        root_dir=os.path.join(VOLUME_PATH, "sorted_data"),
        transform=test_transforms
    )
    
    # Initialize results
    problematic_samples = []
    
    # Check accuracy
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, true_label = dataset[idx]
            image = image.unsqueeze(0).to(device)
            
            # Get prediction and confidence
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_label = torch.max(probabilities, 1)
            confidence = confidence.item()
            pred_label = pred_label.item()
            
            # Check for issues
            if (pred_label != true_label) or (confidence < confidence_threshold):
                sample_path = dataset.image_files[idx]
                problematic_samples.append([
                    os.path.basename(sample_path),
                    true_label,
                    pred_label,
                    confidence
                ])
    
    # Export to CSV
    output_path = os.path.join(VOLUME_PATH, "problematic_samples.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "predicted_label", "confidence"])
        writer.writerows(problematic_samples)
    
    print(f"Found {len(problematic_samples)} problematic samples. Exported to {output_path}")

@app.local_entrypoint()
def main():
    check_accuracy.remote(confidence_threshold=0.9)
