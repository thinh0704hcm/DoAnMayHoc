# Submission ID : 		????
import torch
from PIL import Image
from torchvision import transforms
import os
import csv
import torch.nn as nn

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset class (included for completeness, though not directly used in this prediction script)
class DigitDataset(torch.utils.data.Dataset):
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

# Define transformations (using test_transforms as this is for prediction)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to preprocess a single image
def preprocess_image(image_path, transform):
    """
    Loads and preprocesses a single image.

    Args:
        image_path (str): Path to the image file.
        transform (callable): The transformations to apply to the image.

    Returns:
        torch.Tensor: The processed image tensor, ready for model input.
    """
    try:
        image = Image.open(image_path).convert('RGB') # Open image and convert to RGB
        if transform:
            image = transform(image)
        return image.unsqueeze(0) # Add a batch dimension
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred during image preprocessing: {e}")
        return None

# Function to predict the digit using the trained model
def predict_digit(model, image_tensor, device):
    """
    Uses the trained model to predict the digit in an image tensor.

    Args:
        model (torch.nn.Module): The trained model.
        image_tensor (torch.Tensor): The processed image tensor.
        device (torch.device): The device the model is on (cuda or cpu).

    Returns:
        int: The predicted digit.
    """
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

# Function to build the model architecture
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
        # Import models here to make the function standalone
        import torchvision.models as models
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

# Define the path to the saved model weights
# **IMPORTANT**: Make sure this path is correct and accessible
model_save_path = "/content/handwritten_digit_model_weights.pth"

# Define the folder containing the new images you want to predict
# **IMPORTANT**: Ensure the zip file is in your Google Drive and the path is correct
zip_file_path = "/content/drive/MyDrive/data.2025.zip"
extracted_dir = "./prediction_data" # Directory to extract the images

# Create the extraction directory if it doesn't exist
os.makedirs(extracted_dir, exist_ok=True)

# Extract the zip file
print(f"Extracting data from {zip_file_path} to {extracted_dir}...")
# Use the shell command directly and capture output if needed, or just execute
!unzip -o "$zip_file_path" -d "$extracted_dir"

# After extraction, the images will likely be directly in the extracted_dir or in a subdirectory.
# Inspect the extracted_dir to confirm the actual path to the images.
# Assuming the images are directly in the extracted directory after extraction:
new_images_folder = extracted_dir
print(f"Images to predict from: {new_images_folder}")


# Define the path for the output CSV file
output_csv_path = "predictions.csv"

# Instantiate the model with the same architecture as the trained model
num_classes = 10 # Make sure this matches the number of classes used during training
loaded_model = build_model(num_classes, device) # Use the build_model function to create the model

# Load the saved state dictionary
try:
    loaded_model.load_state_dict(torch.load(model_save_path))
    print(f"Model weights loaded successfully from {model_save_path}")
except FileNotFoundError:
    print(f"Error: Model weights file not found at {model_save_path}. Please ensure the path is correct and the file exists.")
    loaded_model = None # Set model to None if weights couldn't be loaded
except Exception as e:
    print(f"An error occurred while loading model weights: {e}")
    loaded_model = None

if loaded_model is not None:
    loaded_model.to(device) # Move the loaded model to the correct device
    loaded_model.eval() # Set to evaluation mode

    print(f"\nPredicting digits for images in: {new_images_folder}")

    # Open the CSV file for writing
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Iterate through the images in the new images folder
        if os.path.isdir(new_images_folder):
            # Iterate directly through files in the folder, not subfolders
            for image_file in os.listdir(new_images_folder):
                # Construct the full image path
                image_path = os.path.join(new_images_folder, image_file)

                # Check if it's a file and a supported image format
                if os.path.isfile(image_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')) and not image_file.startswith('.'):

                    # Preprocess the image
                    processed_image = preprocess_image(image_path, test_transforms)

                    if processed_image is not None:
                        # Predict the digit
                        predicted_label = predict_digit(loaded_model, processed_image, device)

                        # Write the filename and predicted label to the CSV file
                        writer.writerow([image_file, predicted_label])
                    else:
                        print(f"Could not process image: {image_file}")
        else:
            print(f"Error: The directory '{new_images_folder}' was not found.")

    print(f"Predictions saved to {output_csv_path}")

else:
    print("Model could not be loaded. Prediction skipped.")