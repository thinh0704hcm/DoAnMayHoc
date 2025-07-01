import os
import modal
import math

# --- Modal setup ---
image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1",
    "tensorboard",
    "opencv-python-headless==4.11.0.86",
    "numpy",

)

app = modal.App("pytorch-digit-training", image=image)
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol", create_if_missing=True)
CHECKPOINT_DIR = os.path.join(VOLUME_PATH, "checkpoints/resnet101_processed")
TENSORBOARD_DIR = os.path.join(VOLUME_PATH, "tensorboard")

# --- Training function ---
@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=60*60*12
)
def train():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import cv2
    import numpy as np
    from torchvision import transforms, models
    from torch.utils.data import IterableDataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from PIL import Image
    import os
    import math

    # --- T4-optimized settings ---
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')  # For PyTorch >= 2.0

    num_classes = 10
    num_epochs = 12
    use_amp = True  # Automatic Mixed Precision

    # --- Image Preprocessing Functions ---
    THRESHOLD = 25

    def sharpen_image(image):
        gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=1)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened

    def preprocess_block(image):
        image = cv2.medianBlur(image, 3)
        image = cv2.GaussianBlur(image, (7, 7), 0)
        return 255 - image

    def postprocess_block(image):
        return cv2.medianBlur(image, 5)

    def get_block_index(image_shape, yx, block_size): 
        y = np.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
        x = np.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
        return tuple(np.meshgrid(y, x))

    def adaptive_median_threshold(img_in):
        med = np.median(img_in)
        img_out = np.zeros_like(img_in)
        img_out[img_in - med < THRESHOLD] = 255
        return img_out

    def block_image_process(image, block_size):
        out_image = np.zeros_like(image)
        for row in range(0, image.shape[0], block_size):
            for col in range(0, image.shape[1], block_size):
                idx = (row, col)
                block_idx = get_block_index(image.shape, idx, block_size)
                out_image[block_idx] = adaptive_median_threshold(image[block_idx])
        return out_image

    def remove_lines_hough(binary_image):
        edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=85,
            minLineLength=int(binary_image.shape[1]*0.3),
            maxLineGap=30
        )
        result = binary_image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                if abs(angle) < 15 or abs(angle) > 165 or (75 < abs(angle) < 105):
                    thickness = max(2, int(binary_image.shape[1]/200))
                    cv2.line(result, (x1, y1), (x2, y2), 255, thickness)
        return result

    def preprocess_image(img_bgr):
        """Full preprocessing pipeline for a BGR image"""
        # 1. Sharpen
        sharpened = sharpen_image(img_bgr)
        
        # 2. Convert to LAB and extract luminance
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        
        # 3. Adaptive shadow removal
        width = l_channel.shape[1]
        kernel_size = max(3, int(width / 6))
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)
        normalized = cv2.divide(l_channel, blurred, scale=255)
        
        # 4. Block processing
        block_size_val = max(10, int(width / 10))
        if block_size_val % 2 == 0:
            block_size_val += 1
        preprocessed = preprocess_block(normalized)
        block_processed = block_image_process(preprocessed, block_size_val)
        final_thresh = postprocess_block(block_processed)

        # 5. Line removal
        final_no_lines = remove_lines_hough(final_thresh)
        final_result = remove_lines_hough(final_no_lines)

        return final_result

    # --- Modified Dataset Class ---
    class DigitIterableDataset(IterableDataset):
        def __init__(self, root_dir, transform=None, indices=None):
            self.root_dir = root_dir
            self.transform = transform
            self.indices = indices
            self.image_label_pairs = []
            
            for fname in sorted(os.listdir(root_dir)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')) or fname.startswith('.'):
                    continue
                try:
                    label_str = fname.split('_')[0]
                    label = int(label_str)
                    self.image_label_pairs.append((os.path.join(root_dir, fname), label))
                except Exception:
                    continue
                    
            if indices is not None:
                self.image_label_pairs = [self.image_label_pairs[i] for i in indices]

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            pairs = self.image_label_pairs
            if worker_info is None:
                start, end = 0, len(pairs)
            else:
                per_worker = math.ceil(len(pairs) / worker_info.num_workers)
                worker_id = worker_info.id
                start = worker_id * per_worker
                end = min(start + per_worker, len(pairs))
                
            for idx in range(start, end):
                img_path, label = pairs[idx]
                try:
                    # Load image with OpenCV
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                        
                    # Apply custom preprocessing
                    preprocessed_bgr = preprocess_image(img_bgr)
                    
                    # Convert to RGB for PIL
                    preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(preprocessed_rgb)
                    
                    # Apply standard transforms
                    if self.transform:
                        image = self.transform(image)
                        
                    yield image, label
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue

        def __len__(self):
            return len(self.image_label_pairs)


    # --- Data Transforms ---
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.Resize((224, 224)),  # Ensure consistent input size for channels_last
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- Data Loading ---
    data_root = os.path.join(VOLUME_PATH, "sorted_data")
    full_dataset = DigitIterableDataset(data_root)
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    indices = list(range(total_samples))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = DigitIterableDataset(data_root, train_transforms, train_indices)
    test_dataset = DigitIterableDataset(data_root, test_transforms, test_indices)

    # --- DataLoader with T4-optimized settings ---
    batch_size = 64  # Increase batch size for T4, tune as needed
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # --- Model Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model Setup ---
    original_model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    original_model.fc = nn.Linear(original_model.fc.in_features, num_classes)
    original_model = original_model.to(device, memory_format=torch.channels_last)

    # Compile model để training
    compiled_model = torch.compile(original_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(original_model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # --- TensorBoard & Checkpoint Setup ---
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        original_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from checkpoint at epoch {start_epoch}")

    # --- Training Loop with AMP ---
    for epoch in range(start_epoch, num_epochs):
        compiled_model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda',enabled=use_amp):
                outputs = compiled_model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)
        if len(train_loader) > 0:
            avg_loss = running_loss / len(train_loader)
        else:
            avg_loss = 0.0
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': original_model.state_dict(),  # Quan trọng: lưu từ original model
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        volume.commit()
        torch.cuda.empty_cache()  # Optional: clear cache to avoid fragmentation

    writer.close()
    print("Training completed.")

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
