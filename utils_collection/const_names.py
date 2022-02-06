from pathlib import Path


# Filenames
METADATA_FILE_INPUT = "metadata_input.json"
METADATA_SUBDIR = "data/{}"
DATASET_INFO_FILE = "dataset_info.json"
METADATA_FINAL_FILE = "metadata_final.json"
SEGMENT_LABELS_FILE = "segment_labels.json"

# Train/val/test splits
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"

# Processing status of videos
STATUS_START = "start"
STATUS_UNTAR_DONE = "untar_done"
STATUS_FEATURES_DONE = "features_done"

# set this to "" to download models 
PRETRAINED_MODELS_PATH = Path("pretrained_models")
MODEL_PATH_WORD2VEC = ""
MAPPING_WORD2VEC = {
    "default": "GoogleNews-vectors-negative300"
}

FILETYPES = ["mp4", "mkv", "webm"]
