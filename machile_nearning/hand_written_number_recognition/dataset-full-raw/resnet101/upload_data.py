# Upload script (run locally once)
from modal import Volume

# Get reference to your volume
volume = Volume.from_name("digit-training-vol")

# Upload local dataset directory to volume
# local_data_path = "../dataset-full-raw/sorted_data"

# with volume.batch_upload() as batch:
#     batch.put_directory(local_path=local_data_path, remote_path="/sorted_data")

local_data_path = "./checkpoints"

with volume.batch_upload() as batch:
    batch.put_directory(local_path=local_data_path, remote_path="/saved_checkpoints")
