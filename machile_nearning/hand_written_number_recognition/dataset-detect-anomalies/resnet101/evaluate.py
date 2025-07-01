import modal
import os
import math

image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1"
)

app = modal.App("pytorch-digit-evaluation", image=image)
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol")
CHECKPOINT_DIR = os.path.join(VOLUME_PATH, "checkpoints")

@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=60*60*1
)
def evaluate():
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from torch.utils.data import IterableDataset, DataLoader
    from PIL import Image

    num_classes = 10

    # --- Iterable Dataset Implementation ---
    class DigitIterableDataset(IterableDataset):
        def __init__(self, root_dir, transform=None, indices=None):
            self.root_dir = root_dir
            self.transform = transform
            self.indices = indices
            self.image_label_pairs = []
            for fname in sorted(os.listdir(root_dir)):
                # Bỏ qua file ẩn và không phải ảnh
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')) or fname.startswith('.'):
                    continue
                # Lấy label từ prefix trước dấu '_'
                try:
                    label_str = fname.split('_')[0]
                    label = int(label_str)
                except Exception:
                    continue
                self.image_label_pairs.append(
                    (os.path.join(root_dir, fname), label)
                )
            if indices is not None:
                self.image_label_pairs = [self.image_label_pairs[i] for i in indices]

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            pairs = self.image_label_pairs
            
            if worker_info is None:  # Single-process
                start, end = 0, len(pairs)
            else:  # Multi-worker
                per_worker = math.ceil(len(pairs) / worker_info.num_workers)
                worker_id = worker_info.id
                start = worker_id * per_worker
                end = min(start + per_worker, len(pairs))

            for idx in range(start, end):
                img_path, label = pairs[idx]
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                yield image, label

        def __len__(self):
            return len(self.image_label_pairs)

    # --- Evaluation Function ---
    def evaluate_model(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        if total_samples == 0:
            raise ValueError("No samples processed during evaluation")
            
        average_loss = running_loss / total_samples
        accuracy = correct_predictions / total_samples
        return average_loss, accuracy

    # --- Main Logic ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo model gốc
    model = models.resnet101(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device, memory_format=torch.channels_last)
    
    # Load checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load trực tiếp không cần xử lý prefix
    
    # Compile model để evaluation
    compiled_model = torch.compile(model)  # Luôn compile

    # Test transforms (matches training)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Create datasets
    data_root = os.path.join(VOLUME_PATH, "processed_data")

    # 1. Create full dataset first
    full_dataset = DigitIterableDataset(data_root, transform=test_transforms)
    if len(full_dataset) == 0:
        raise ValueError(f"No data found in {data_root}. Check volume contents.")

    # 2. Create test dataset
    total_samples = len(full_dataset)
    test_indices = list(range(int(0.8 * total_samples), total_samples))
    test_dataset = DigitIterableDataset(
        root_dir=data_root,
        transform=test_transforms,
        indices=test_indices
    )

    # 3. Validate test dataset
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Check split ratio or data indices.")

    # Create test loader (matches training config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Run evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate_model(compiled_model, test_loader, criterion, device)
    
    print("\n" + "="*50)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print("="*50 + "\n")

@app.local_entrypoint()
def main():
    evaluate.remote()
