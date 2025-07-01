import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
import shutil

# === CONFIGURATION ===
PROBLEMATIC_CSV = "problematic_samples.csv"       # Path to your problematic samples CSV
DATA_ROOT = "../dataset-full-raw/sorted_data"                         # Root directory containing label folders (e.g., sorted_data/0, sorted_data/1, ...)
REVIEWED_CSV = "manual_review_results.csv"        # Output CSV for review results

def find_image_path(filename):
    """Search all label folders for the file."""
    for label_folder in os.listdir(DATA_ROOT):
        folder_path = os.path.join(DATA_ROOT, label_folder)
        if os.path.isdir(folder_path):
            candidate = os.path.join(folder_path, filename)
            if os.path.exists(candidate):
                return candidate, label_folder
    return None, None

def review():
    with open(PROBLEMATIC_CSV, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        samples = list(reader)

    reviewed = []
    for sample in samples:
        filename = sample['filename']
        true_label = sample.get('true_label', '')
        predicted_label = sample.get('predicted_label', '')
        confidence = sample.get('confidence', '')

        image_path, current_label = find_image_path(filename)
        if not image_path:
            print(f"Image {filename} not found, skipping.")
            continue

        # Display image and info
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{filename}\nTrue: {true_label} | Pred: {predicted_label} | Conf: {confidence}")
        plt.show(block=False)
        plt.pause(0.001)  # Allows the window to appear

        input("Press Enter after reviewing the image window...")  # Now you can interact with the terminal

        plt.close()  # Close the image window after input


        # Prompt for manual review
        print(f"\nFile: {filename}")
        print(f"Current folder (label): {current_label}")
        print(f"True label: {true_label}")
        print(f"Predicted label: {predicted_label} (Confidence: {confidence})")
        print("Options: [Enter] keep label, [0-9] new label, [d] delete sample")
        user_input = input("Enter your choice: ").strip()

        if user_input.lower() == 'd':
            # Delete the sample
            os.remove(image_path)
            print(f"Deleted {filename}")
            reviewed.append({
                "filename": filename,
                "old_label": current_label,
                "action": "deleted",
                "new_label": "",
                "predicted_label": predicted_label,
                "confidence": confidence
            })
            continue

        if user_input == "":
            # Keep current label
            new_label = current_label
            action = "kept"
        elif user_input.isdigit() and user_input in [str(i) for i in range(10)]:
            new_label = user_input
            if new_label != current_label:
                # Move file to new label folder
                new_folder = os.path.join(DATA_ROOT, new_label)
                os.makedirs(new_folder, exist_ok=True)
                new_path = os.path.join(new_folder, filename)
                shutil.move(image_path, new_path)
                print(f"Moved {filename} from {current_label} to {new_label}")
                action = "moved"
            else:
                print("Label unchanged.")
                action = "kept"
        else:
            print("Invalid input, keeping current label.")
            new_label = current_label
            action = "kept"

        reviewed.append({
            "filename": filename,
            "old_label": current_label,
            "action": action,
            "new_label": new_label,
            "predicted_label": predicted_label,
            "confidence": confidence
        })

    # Save review results
    with open(REVIEWED_CSV, "w", newline='') as csvfile:
        fieldnames = ["filename", "old_label", "action", "new_label", "predicted_label", "confidence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reviewed:
            writer.writerow(row)

    print(f"Manual review completed. Results saved to {REVIEWED_CSV}")

if __name__ == "__main__":
    review()
