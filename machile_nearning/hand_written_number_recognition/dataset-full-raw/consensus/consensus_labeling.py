# Submission ID :    272804
import modal
import os

image = modal.Image.debian_slim().pip_install(
    "torch==2.7.1",
    "torchvision==0.22.1",
    "efficientnet_pytorch",
    "timm",
    "pillow"
)

app = modal.App("consensus-labeling", image=image)
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol")

@app.function(
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=60*60,  # 1 hour
)
def generate_consensus_labels():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    import timm
    import torchvision.models as models
    from efficientnet_pytorch import EfficientNet
    import csv
    import numpy as np

    # --- Configuration ---
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    CONFIDENCE_THRESHOLD = 0.8
    NUM_CLASSES = 10
    MODEL_PATHS = {
        'resnet50': '/vol/checkpoints_resnet50/latest.pth',
        'resnet101': '/vol/checkpoints/latest.pt',
        'efficientnet_b4': '/vol/checkpoints_efficientnet/latest.pt',
        'vit_base': '/vol/checkpoints_vit/latest.pt'
    }

    # --- Transforms ---
    resnet_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    vit_transform = resnet_transform
    effnet_transform = transforms.Compose([
        transforms.Resize((380,380)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- Dataset ---
    class ConsensusDataset(Dataset):
        def __init__(self, root_dir):
            self.image_paths = []
            self.labels = []
            for label in os.listdir(root_dir):
                label_dir = os.path.join(root_dir, label)
                if os.path.isdir(label_dir):
                    for fname in os.listdir(label_dir):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(label_dir, fname))
                            self.labels.append(int(label))

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            return img, self.labels[idx], os.path.basename(self.image_paths[idx])

    # --- Model Loading ---
    def load_model(model_name):
        device = torch.device("cuda")
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'efficientnet_b4':
            model = EfficientNet.from_name('efficientnet-b4', num_classes=NUM_CLASSES)
        elif model_name == 'vit_base':
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        if model_name == 'resnet50':
            model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location=device))
        else:
            model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location=device)['model_state_dict'])
        model.eval()
        model.to(device)
        return model

    device = torch.device("cuda")
    dataset = ConsensusDataset(os.path.join(VOLUME_PATH, "sorted_data"))

    models_dict = {
        'resnet50': load_model('resnet50'),
        'resnet101': load_model('resnet101'),
        'efficientnet_b4': load_model('efficientnet_b4'),
        'vit_base': load_model('vit_base')
    }

    # --- DataLoader with Efficient Batch Collation ---
    def collate_fn(batch):
        images, labels, fnames = zip(*batch)
        return (
            [resnet_transform(img) for img in images],
            [effnet_transform(img) for img in images],
            [vit_transform(img) for img in images],
            torch.tensor(labels),
            fnames
        )

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # --- Inference Loop with Mixed Precision ---
    with (
    open(os.path.join(VOLUME_PATH, "consensus.csv"), 'w', newline='') as csv_file,
    open(os.path.join(VOLUME_PATH, "flagged.csv"), 'w', newline='') as flag_file
):
        csv_writer = csv.writer(csv_file)
        flag_writer = csv.writer(flag_file)
        csv_writer.writerow(['Filename', 'TrueLabel', 'ResNet50', 'ResNet101', 'EffNetB4', 'ViTBase', 'Consensus'])
        flag_writer.writerow(['Filename', 'Reason'])

        for resnet_imgs, effnet_imgs, vit_imgs, labels, fnames in dataloader:
            resnet_batch = torch.stack(resnet_imgs).to(device, non_blocking=True)
            effnet_batch = torch.stack(effnet_imgs).to(device, non_blocking=True)
            vit_batch = torch.stack(vit_imgs).to(device, non_blocking=True)

            with torch.no_grad(), torch.amp.autocast('cuda'):
                preds_resnet50 = models_dict['resnet50'](resnet_batch).softmax(1)
                preds_resnet101 = models_dict['resnet101'](resnet_batch).softmax(1)
                preds_effnet = models_dict['efficientnet_b4'](effnet_batch).softmax(1)
                preds_vit = models_dict['vit_base'](vit_batch).softmax(1)

                for i in range(len(labels)):
                    fname = fnames[i]
                    true_label = labels[i].item()
                    pred50 = preds_resnet50[i].argmax().item()
                    conf50 = preds_resnet50[i].max().item()
                    pred101 = preds_resnet101[i].argmax().item()
                    conf101 = preds_resnet101[i].max().item()
                    pred_b4 = preds_effnet[i].argmax().item()
                    conf_b4 = preds_effnet[i].max().item()
                    pred_vit = preds_vit[i].argmax().item()
                    conf_vit = preds_vit[i].max().item()
                    model_preds = [pred50, pred101, pred_b4, pred_vit]
                    confidences = [conf50, conf101, conf_b4, conf_vit]
                    counts = np.bincount(model_preds, minlength=NUM_CLASSES)
                    consensus = np.argmax(counts)
                    consensus_strength = np.max(counts) / 4

                    csv_writer.writerow([fname, true_label, pred50, pred101, pred_b4, pred_vit, consensus])

                    # Flag for review if consensus is weak or confidence is low
                    if consensus_strength < 0.5 or (consensus != true_label and true_label != -1) or min(confidences) < CONFIDENCE_THRESHOLD:
                        flag_writer.writerow([
                            fname,
                            "low_consensus" if consensus_strength < 0.5 else ("label_mismatch" if consensus != true_label else "low_confidence")
                        ])

                # Explicit memory cleanup for large batches
                del resnet_batch, effnet_batch, vit_batch
                del preds_resnet50, preds_resnet101, preds_effnet, preds_vit
                torch.cuda.empty_cache()

    print("Consensus labeling complete!")

@app.local_entrypoint()
def main():
    generate_consensus_labels.remote()
