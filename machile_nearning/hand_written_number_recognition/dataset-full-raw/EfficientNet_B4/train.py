# Submission ID :          274378
import modal
import os

image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1",
    "efficientnet_pytorch",
    "tensorboard"
)

app = modal.App("efficientnet-b4-training", image=image)

VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol", create_if_missing=True)
CHECKPOINT_DIR = os.path.join(VOLUME_PATH, "checkpoints_efficientnet")
TENSORBOARD_DIR = os.path.join(VOLUME_PATH, "tensorboard_efficientnet")

@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=60*60*12
)
def train():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from efficientnet_pytorch import EfficientNet
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader, random_split, Subset
    from torch.utils.tensorboard import SummaryWriter
    from torch.cuda.amp import autocast, GradScaler
    from PIL import Image
    import os

    # --- Dataset ---
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

    # --- Config ---
    num_classes = 10
    num_epochs = 12
    batch_size = 16  # Reduced from 24
    test_batch_size = 64  # Reduced from 128
    input_size = (380, 380)

    # --- Data ---
    train_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=90, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_root = os.path.join(VOLUME_PATH, "sorted_data")
    full_dataset = DigitDataset(root_dir=data_root, transform=None)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = random_split(range(len(full_dataset)), [train_size, test_size])
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    train_dataset.dataset.transform = train_transforms
    test_dataset.dataset.transform = test_transforms
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # --- Cài đặt model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_pretrained("efficientnet-b4", num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

    # --- Data ---
    train_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=45, translate=(0.1, 0.1)),  # Reduced rotation
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),  # Reduced distortion
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # --- DataLoader optimization ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,  # Faster data transfer to GPU
        num_workers=2     # Parallel data loading
    )
    
    # --- Mixed Precision Setup ---
    scaler = GradScaler()

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Mixed precision context
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Scaled backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            
            # Memory cleanup
            del inputs, labels, outputs
            torch.cuda.empty_cache()
        avg_train_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # --- Evaluation ---
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del inputs, labels, outputs
                torch.cuda.empty_cache()
        avg_val_loss = val_loss / len(test_loader.dataset)
        val_acc = correct / total
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        print(f"Epoch {epoch+1}: TrainLoss={avg_train_loss:.4f} ValLoss={avg_val_loss:.4f} ValAcc={val_acc:.4f}")

        # --- Save checkpoint ---
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(CHECKPOINT_DIR, "latest.pt"))
        volume.commit()
    writer.close()
    print("EfficientNet-B4 training complete.")

@app.function(
    volumes={VOLUME_PATH: volume},
    max_containers=1,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def tensorboard_app():
    import tensorboard
    board = tensorboard.program.TensorBoard()
    board.configure(logdir=TENSORBOARD_DIR)
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
        experimental_middlewares=[VolumeMiddleware],
    )
    return wsgi_app

# --- Local entrypoint ---
@app.local_entrypoint()
def main():
    train.remote()
    print("Training started. TensorBoard will be available at the Modal-provided URL.")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating app.")
