import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import pillow_heif

COMBINED_DIR = "combined_data"
SORTED_DIR = "sorted_data_augmented"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".heic")
ROT_ANGLES_SPECIAL = [0, 30, 60]
ROT_ANGLES_DEFAULT = [0, 60, 120, 180, 240, 300]
SPECIAL_DIGITS = {'6', '8', '9', '0'}

os.makedirs(SORTED_DIR, exist_ok=True)
pillow_heif.register_heif_opener()  # Register once

def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".heic":
        img = Image.open(path).convert("RGB")
        jpg_path = path.rsplit('.', 1)[0] + ".jpg"
        if not os.path.exists(jpg_path):
            img.save(jpg_path, "JPEG", quality=95)
        return img, jpg_path
    else:
        img = Image.open(path).convert("RGB")
        return img, path

def augment_and_save(fname):
    try:
        label = fname[0]
        if not label.isdigit():
            return 0, 1  # SL đã chỉnh sửa, flag bỏ qua

        label_dir = os.path.join(SORTED_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        src = os.path.join(COMBINED_DIR, fname)

        img, used_path = load_image(src)
        base_name = os.path.splitext(os.path.basename(used_path))[0]

        # Chọn góc xoay dựa trên nhãn
        if label in SPECIAL_DIGITS:
            angles = ROT_ANGLES_SPECIAL
        else:
            angles = ROT_ANGLES_DEFAULT

        count = 0
        for angle in angles:
            rotated = img.rotate(angle, expand=True)
            out_fname = f"{base_name}_rot{angle}.jpg"
            out_path = os.path.join(label_dir, out_fname)
            rotated.save(out_path, "JPEG", quality=95)
            count += 1
        return count, 0 # Số lượng ảnh đã xoay, flag bỏ qua
    except Exception as e:
        print(f"Error with {fname}: {e}")
        return 0, 1

if __name__ == "__main__":
    files = [f for f in os.listdir(COMBINED_DIR)
             if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(COMBINED_DIR, f))]
    print(f"Found {len(files)} images in {COMBINED_DIR}")

    augmented = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = {executor.submit(augment_and_save, fname): fname for fname in files}
        for future in as_completed(futures):
            a, s = future.result()
            augmented += a
            skipped += s

    print(f"Augmented and saved {augmented} images into {SORTED_DIR}/<digit>/ folders.")
    if skipped:
        print(f"Skipped {skipped} files (non-digit prefix or error).")
