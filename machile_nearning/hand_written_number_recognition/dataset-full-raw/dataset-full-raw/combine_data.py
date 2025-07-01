import os
import glob
import shutil
import uuid

# 1. Define paths
combined_dir = "./combined_data"
os.makedirs(combined_dir, exist_ok=True)

# 2. Find all image paths
a = glob.glob(f'./github_cloned/*/hand_written_digit/??52????')
image_lists = []
for folder in a:
    for num in range(10):
        t = glob.glob(f'{folder}/{num}_*')
        image_lists += t

# 3. Copy and rename files with UUID
file_count = 0
for img_path in image_lists:
    try:
        # Get original filename parts
        filename = os.path.basename(img_path)
        label = filename[0]  # First character is the label
        ext = os.path.splitext(filename)[1]  # Keep original extension
        
        # Generate unique filename with label and UUID
        new_filename = f"{label}_{uuid.uuid4()}{ext}"
        dest_path = os.path.join(combined_dir, new_filename)
        
        # Copy file to new location
        shutil.copy2(img_path, dest_path)
        file_count += 1
    except Exception as e:
        print(f"Failed to process {img_path}: {str(e)}")

print(f"\nSuccessfully combined {file_count} files into {combined_dir}")
