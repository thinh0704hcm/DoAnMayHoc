import modal
import os

# Modal setup
image = modal.Image.debian_slim().pip_install("zipfile36")  # zipfile is in stdlib, pip_install not strictly needed
app = modal.App("extract-sorted-data", image=image)

VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("digit-training-vol", create_if_missing=True)

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 10  # 10 minutes should be enough
)
def extract_zip():
    import zipfile

    zip_path = os.path.join(VOLUME_PATH, "sorted_data.zip")
    extract_to = VOLUME_PATH

    if not os.path.exists(zip_path):
        print(f"Zip file not found at {zip_path}")
        return

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
