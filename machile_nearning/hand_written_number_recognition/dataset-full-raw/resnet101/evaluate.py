# Submission ID :       272723
import modal
import os

# Sử dụng image và volume giống với training
image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1"
)

app = modal.App("pytorch-digit-evaluation", image=image)
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol")
CHECKPOINT_DIR = os.path.join(VOLUME_PATH, "checkpoints")

# --- Evaluation Function ---
@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=60*60*1  # 1 hour timeout
)
def evaluate():
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from torch.utils.data import Dataset, DataLoader, Subset
    from PIL import Image

    # --- Dataset Class (giống code gốc) ---
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
                        continue

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    # --- Hàm evaluation (giống code gốc) ---
    def evaluate_model(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        average_loss = running_loss / total_samples
        accuracy = correct_predictions / total_samples
        return average_loss, accuracy

    # --- Main Evaluation Logic ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    num_classes = 10
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load checkpoint từ volume
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    model.load_state_dict((torch.load(checkpoint_path, map_location=device))['model_state_dict'])
    model = model.to(device)
    
    # Chuẩn bị test loader
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    data_root = os.path.join(VOLUME_PATH, "sorted_data")
    test_loader = DataLoader(
        Subset(
            DigitDataset(root_dir=data_root, transform=None),
            indices=list(range(int(0.8*len(DigitDataset(data_root))), len(DigitDataset(data_root))))
        ),
        batch_size=1000,
        shuffle=False
    )
    test_loader.dataset.dataset.transform = test_transforms

    # Chạy evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    
    print("\n" + "="*50)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print("="*50 + "\n")

# --- Local Entrypoint ---
@app.local_entrypoint()
def main():
    evaluate.remote()
