# Submission ID : 		272566
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image, ImageOps, ImageFilter
import torch.optim as optim

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the root directory where the data was extracted
data_root = "/content/sorted_data"

def clean_image(image):
    # Convert to grayscale for thresholding
    gray = image.convert('L')
    # Adaptive thresholding
    bw = gray.point(lambda x: 0 if x < 128 else 255, '1')
    # Auto-crop to content
    bbox = bw.getbbox()
    if bbox:
        cropped = image.crop(bbox)
    else:
        cropped = image
    # Add padding to restore to square
    padded = ImageOps.pad(cropped, (224, 224), color=(255,255,255))
    return padded

# Custom Dataset class
class DigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the digit folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Walk through the directory to find image files and their labels
        for label_folder in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_folder)
            if os.path.isdir(label_path):
                try:
                    label = int(label_folder) # Assuming folder names are the digit labels (0-9)
                    for image_file in os.listdir(label_path):
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')): # Add more image formats if needed and handle case
                             # Check if the file is not a hidden file
                            if not image_file.startswith('.'):
                                self.image_files.append(os.path.join(label_path, image_file))
                                self.labels.append(label)
                except ValueError:
                    print(f"Skipping non-integer folder: {label_folder}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB') # Open image and convert to RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_and_split_data(data_root, train_transforms, test_transforms, train_split=0.8, batch_size=64, test_batch_size=1000):
    """
    Loads the dataset, splits it into training and testing sets, and creates DataLoaders.

    Args:
        data_root (str): Directory with all the digit folders.
        train_transforms (callable): Transformations for the training set.
        test_transforms (callable): Transformations for the testing set.
        train_split (float): The proportion of the dataset to use for training.
        batch_size (int): Batch size for the training loader.
        test_batch_size (int): Batch size for the testing loader.

    Returns:
        tuple: A tuple containing train_loader and test_loader.
    """
    try:
        full_dataset = DigitDataset(root_dir=data_root, transform=None)

        train_size = int(train_split * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_indices, test_indices = random_split(range(len(full_dataset)), [train_size, test_size])

        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # Apply transforms to the subsets
        train_dataset.dataset.transform = train_transforms
        test_dataset.dataset.transform = test_transforms

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        print("Dataset loaded, split, and DataLoaders created.")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of testing samples: {len(test_dataset)}")

        return train_loader, test_loader

    except FileNotFoundError:
        print(f"Error: The directory '{data_root}' was not found. Please check the path to your extracted data.")
        return None, None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None, None

def build_model(num_classes, device):
    """
    Builds the transfer learning model by loading a pre-trained model and modifying its final layer.

    Args:
        num_classes (int): The number of output classes.
        device (torch.device): The device to load the model onto.

    Returns:
        torch.nn.Module: The configured model.
    """
    try:
        model = models.resnet50(pretrained=True)
        print("Pre-trained ResNet50 model loaded.")

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print(f"Modified the final layer for {num_classes} classes.")

        model = model.to(device)
        print("Model moved to device.")

        return model
    except Exception as e:
        print(f"An error occurred during model building: {e}")
        return None


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model's performance on the given dataset.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to evaluate on.

    Returns:
        tuple: A tuple containing the average loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation
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


def train(model, train_loader, test_loader, optimizer, criterion, num_epochs, device, model_save_path="handwritten_digit_model_weights.pth", optimizer_save_path="handwritten_digit_optimizer_state.pth"):
    """
    Trains the model and evaluates it after training.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        num_epochs (int): The number of epochs to train for.
        device (torch.device): The device to train on (cuda or cpu).
        model_save_path (str): Path to save the model weights.
        optimizer_save_path (str): Path to save the optimizer state.
    """
    model.train() # Set the model to training mode
    print("Starting training...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (i + 1) % 100 == 0: # Print every 100 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {running_loss / len(train_loader):.4f}')

    try:
        # Save the model's state dictionary
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")

        # Save the optimizer's state dictionary
        torch.save(optimizer.state_dict(), optimizer_save_path)
        print(f"Optimizer state saved to {optimizer_save_path}")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

    print("Training finished.")

def evaluate(model, dataloader, criterion, device):
    print("Evaluating model on the test set...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


# --- Main Execution ---

# Define transformations
train_transforms = transforms.Compose([
    transforms.Lambda(clean_image),  # Clean background, crop, pad, ensure RGB
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(
        degrees=15,  # Rotation already handled above
        translate=(0.1, 0.1),  # Allow up to 10% translation in both directions
        scale=(0.9, 1.1)  # Minor scaling for size variation
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Simulate viewpoint changes
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB normalization
])

test_transforms = transforms.Compose([
    transforms.Lambda(clean_image),  # Essential - same cleaning as training
    transforms.Resize((224, 224)),   # Optional safety (clean_image already pads to 224x224)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Define the path to the saved model weights
model_save_path = "handwritten_digit_model_weights.pth"
optimizer_save_path = "handwritten_digit_optimizer_state.pth"

# Load and split data
train_loader, test_loader = load_and_split_data(data_root, train_transforms, test_transforms, 0.8, 32)

if train_loader is not None and test_loader is not None:
    # Build the model
    num_classes = 10
    model = build_model(num_classes, device)

    if model is not None:
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Check if saved weights exist and load them
        if os.path.exists(model_save_path):
            try:
                model.load_state_dict(torch.load(model_save_path))
                print(f"Loaded model weights from {model_save_path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")


        if os.path.exists(optimizer_save_path):
            try:
                optimizer.load_state_dict(torch.load(optimizer_save_path))
                print(f"Loaded optimizer state from {optimizer_save_path}")

            except Exception as e:
                 print(f"Error loading optimizer state: {e}")


        # Train and evaluate the model
        num_epochs = 12 # You can adjust the number of epochs
        train(model, train_loader, test_loader, optimizer, criterion, num_epochs, device, model_save_path=model_save_path, optimizer_save_path=optimizer_save_path)
        evaluate(model, test_loader, criterion, device)

    else:
        print("Model building failed. Training skipped."