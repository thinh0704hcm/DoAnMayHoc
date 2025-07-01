# Upload script (run locally once)
from modal import Volume

# Get reference to your volume
volume = Volume.from_name("digit-training-vol")

# Upload local dataset directory to volume
local_data_path = "../dataset-detect-anomalies/step3_grids_removed"

with volume.batch_upload() as batch:
    batch.put_directory(local_path=local_data_path, remote_path="/processed_data")

# local_data_path = "C:/Users/ASUS/Downloads/data.2025"

# with volume.batch_upload() as batch:
#     batch.put_directory(local_path=local_data_path, remote_path="/evaluation_data")
