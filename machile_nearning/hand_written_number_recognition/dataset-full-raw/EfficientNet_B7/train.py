# Submission ID :       274379
import os
import modal

image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1",
    "efficientnet_pytorch",
    "tensorboard"
)

app = modal.App("efficientnet-b7-training", image=image)

VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol", create_if_missing=True)
CHECKPOINT_DIR = os.path.join(VOLUME_PATH, "checkpoints_efficientnet_b7_v2")
TENSORBOARD_DIR = os.path.join(VOLUME_PATH, "tensorboard_efficientnet_b7_v2")

@app.function(
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=60*60*24,  # 24 hour timeout
)
def train_efficientnet_b7():
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from efficientnet_pytorch import EfficientNet
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader, random_split, Subset
    from torch.utils.tensorboard import SummaryWriter
    from torch.cuda.amp import autocast, GradScaler
    from PIL import Image

    # --- Dataset Class (same as B4 implementation) ---
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
    num_epochs = 15
    batch_size = 10  # Reduced for A10G 24GB VRAM
    input_size = (600, 600)
    data_root = os.path.join(VOLUME_PATH, "sorted_data")  # Same dataset as B4

    # --- Transforms (B7-specific) ---
    train_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Prepare Data (same split method as B4) ---
    full_dataset = DigitDataset(root_dir=data_root, transform=None)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = random_split(range(len(full_dataset)), [train_size, test_size])
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    train_dataset.dataset.transform = train_transforms
    test_dataset.dataset.transform = test_transforms

    # --- DataLoader with B7 optimizations ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

    # --- Model Setup ---
    device = torch.device("cuda")
    model = EfficientNet.from_pretrained("efficientnet-b7", num_classes=num_classes)
    model = model.to(device)
    
    # --- Optimizer and Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=2
    )
    
    # --- Training Setup ---
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    scaler = GradScaler()
    best_acc = 0.0

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
        
        avg_train_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        
        # --- Validation ---
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(test_loader.dataset)
        val_acc = correct / total
        scheduler.step(val_acc)
        
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        # --- Save Best Checkpoint ---
        if val_acc > best_acc:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            volume.commit()  # Persist to volume
    
    writer.close()
    print(f"Training complete. Best accuracy: {best_acc:.4f}")

# --- Local Entrypoint ---
@app.local_entrypoint()
def main():
    train_efficientnet_b7.remote()
