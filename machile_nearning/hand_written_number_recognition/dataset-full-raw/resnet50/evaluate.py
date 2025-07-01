# Submission ID : 		272612
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Custom Dataset ---
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

# --- Test Data Transforms ---
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Data Loader ---
def get_test_loader(data_root, test_transforms, test_split=0.2, batch_size=1000):
    full_dataset = DigitDataset(root_dir=data_root, transform=None)
    test_size = int(test_split * len(full_dataset))
    train_size = len(full_dataset) - test_size
    # Use the last part as test set for reproducibility
    indices = list(range(len(full_dataset)))
    test_indices = indices[-test_size:]
    test_dataset = Subset(full_dataset, test_indices)
    test_dataset.dataset.transform = test_transforms
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# --- Model Builder ---
def build_model(num_classes, device):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    return model

# --- Evaluation Function ---
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

# --- Main ---
if __name__ == "__main__":
    # Paths
    data_root = "/content/sorted_data_extracted/sorted_data"  # Update if needed
    model_save_path = "handwritten_digit_model_weights.pth"

    # Prepare test loader
    test_loader = get_test_loader(data_root, test_transforms, test_split=0.2, batch_size=1000)

    # Build and load model
    num_classes = 10
    model = build_model(num_classes, device)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded model weights from {model_save_path}")
    else:
        print(f"Model weights file '{model_save_path}' not found.")
        exit(1)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
