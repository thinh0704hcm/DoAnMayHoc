import modal
import os
import zipfile

# Modal setup
image = modal.Image.debian_slim().pip_install("zipfile36")  # Not strictly needed, zipfile is stdlib
app = modal.App("upload-and-extract-sorted-data", image=image)

VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol", create_if_missing=True)
LOCAL_ZIP = ["sorted_data_augmented.zip", "sorted_data_augmented_2.zip", "sorted_data_augmented_3.zip"]

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 10  # 10 minutes
)
def extract_zip():
    import zipfile
    for zip_file in LOCAL_ZIP:
        zip_path = os.path.join(VOLUME_PATH, zip_file)
        if not os.path.exists(zip_path):
            print(f"Zip file {zip_file} not found at {zip_path}")
            continue

        extract_to = os.path.join(VOLUME_PATH, "sorted_data_augmented")
        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted {zip_path} to {extract_to}")

    # Commit changes to the Modal volume
    volume.commit()
    print("Extraction complete and committed.")

@app.local_entrypoint()
def main():
    extract_zip.remote()
