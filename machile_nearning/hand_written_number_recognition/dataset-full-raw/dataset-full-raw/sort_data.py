import os
import shutil

# Set your combined data folder and output folder
COMBINED_DIR = "data\combined_data"   # Folder with all images (e.g., 3_xxx.png, 8_yyy.jpg, etc.)
SORTED_DIR = "data\sorted_data"       # Output folder (will be created if it doesn't exist)

# Supported image extensions
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

os.makedirs(SORTED_DIR, exist_ok=True)

files = [f for f in os.listdir(COMBINED_DIR) if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(COMBINED_DIR, f))]
print(f"Found {len(files)} images in {COMBINED_DIR}")

moved = 0
skipped = 0
for fname in files:
    try:
        label = fname[0]
        if not label.isdigit():
            skipped += 1
            continue
        label_dir = os.path.join(SORTED_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        src = os.path.join(COMBINED_DIR, fname)
        dst = os.path.join(label_dir, fname)
        shutil.move(src, dst)
        moved += 1
    except Exception as e:
        print(f"Error with {fname}: {e}")
        skipped += 1

print(f"Moved {moved} images into {SORTED_DIR}/<digit>/ folders.")
if skipped:
    print(f"Skipped {skipped} files (non-digit prefix or error).")
