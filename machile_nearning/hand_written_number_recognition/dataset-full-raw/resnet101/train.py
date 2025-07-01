# Submission ID :       272723
import os
import modal

# --- Modal setup ---
# Create a Modal image with PyTorch and TensorBoard
image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1",
    "tensorboard"
)

app = modal.App("pytorch-digit-training", image=image)

# Modal Volume for checkpoints and logs
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol", create_if_missing=True)
CHECKPOINT_DIR = os.path.join(VOLUME_PATH, "checkpoints")
TENSORBOARD_DIR = os.path.join(VOLUME_PATH, "tensorboard")

# --- Training function ---
@app.function(
    gpu="T4",  # or T4, H100, etc.
    volumes={VOLUME_PATH: volume},
    timeout=60*60*12  # 12h, adjust as needed
)
def train():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader, random_split, Subset
    from torch.utils.tensorboard import SummaryWriter
    from PIL import Image
    import os

    num_classes = 10
    num_epochs = 12

    # --- Dataset class (from your code) ---
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

    # --- Data transforms ---
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- Data loading ---
    data_root = os.path.join(VOLUME_PATH, "sorted_data")
    full_dataset = DigitDataset(root_dir=data_root, transform=None)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = random_split(range(len(full_dataset)), [train_size, test_size])
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    train_dataset.dataset.transform = train_transforms
    test_dataset.dataset.transform = test_transforms
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # --- Cài đặt model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

    # --- Checkpoint resume ---
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from checkpoint at epoch {start_epoch}")

    # --- Training loop ---
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Log to TensorBoard
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        volume.commit()  # Persist to Modal volume

    writer.close()
    print("Training finished.")

# --- TensorBoard server ---
class VolumeMiddleware:
    def __init__(self, app):
        self.app = app
    def __call__(self, environ, start_response):
        if (route := environ.get("PATH_INFO")) in ["/", "/modal-volume-reload"]:
            try:
                volume.reload()
            except Exception as e:
                print("Exception while re-loading traces: ", e)
            if route == "/modal-volume-reload":
                environ["PATH_INFO"] = "/"
        return self.app(environ, start_response)

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
