import os

DATA_ROOT = "sorted_data"  # Change if your folder is elsewhere
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')

def main():
    for label in os.listdir(DATA_ROOT):
        label_path = os.path.join(DATA_ROOT, label)
        if not os.path.isdir(label_path):
            continue

        files = [f for f in os.listdir(label_path) if f.lower().endswith(VALID_EXTENSIONS) and not f.startswith('.')]
        files.sort()
        for idx, filename in enumerate(files, start=1):
            ext = os.path.splitext(filename)[1]
            new_name = f"{label}_{idx:05d}{ext}"
            old_path = os.path.join(label_path, filename)
            new_path = os.path.join(label_path, new_name)
            if old_path != new_path:
                # Avoid overwriting existing files
                if os.path.exists(new_path):
                    print(f"Skipping {old_path}, {new_path} already exists!")
                    continue
                os.rename(old_path, new_path)
                print(f"Renamed {old_path} -> {new_path}")

if __name__ == "__main__":
    main()
