"""
Define the datasets here, all scripts will use these parameters unless
otherwise specified.
"""
import json
import os
from pathlib import Path
import socket
from utils_collection import const_names

DATASETS = {
    "precomp_anet": {
        "comment": "precomputed icep features for activitynet",
        "width": -1,
        "height": -1,
        "fps": -1,
        "feature_dim": 2048,
        "default_path": "",
        "local_path": "",
        "splits": "train,val_1,val_2",
        # "splits": "train,val_1",  # ignore val_2 by default
        "content": "features"
    },
    "precomp_tvr": {
        "comment": "precomputed features for tvr",
        "width": -1,
        "height": -1,
        "fps": -1,
        "feature_dim": 3072,
        "default_path": "",
        "local_path": "",
        "splits": "train,val",  # ignore test by default
        "content": "features"
    },
    "precomp_youcook2": {
        "comment": "resnet features",
        "width": -1,
        "height": -1,
        "fps": -1,
        # "feature_dim": 0,  # depends
        "default_path": "/home/ubuntu/workspace/experiments/h100m_youcook2",
        "local_path": "/home/ubuntu/workspace/experiments/h100m_youcook2",
        "video_path": "/mnt/efs/fs1/workspace/datasets/youcook2_mp4",
        "splits": "train,val",  # ignore test by default
        "content": "features"
    },
    "precomp_lsmdc16": {
        "comment": "resnet features",
        "width": -1,
        "height": -1,
        "fps": -1,
        # "feature_dim": 0,  # depends
        "default_path": "/mnt/efs/fs1/workspace/experiments/h100m_lsmdc",
        "local_path": "/mnt/efs/fs1/workspace/experiments/h100m_lsmdc",
        "video_path": "/mnt/efs/fs1/workspace/datasets/lsmdc16_mp4",
        "splits": "train,val",  # ignore test by default
        "content": "features"
    }
}


def get_path(dataset_name, dataset_path=""):
    """ get dataset main path from argparse _results"""
    if dataset_path == "":
        if socket.gethostname().lower() == "dpc":
            dataset_path = DATASETS[dataset_name]["local_path"]
        else:
            dataset_path = DATASETS[dataset_name]["default_path"]
    dataset_path = Path(dataset_path)
    video_path = Path(DATASETS[dataset_name]["video_path"])
    return dataset_path, video_path


def get_splits(dataset_name, splits=""):
    """ get dataset splits from argparse _results"""
    if splits == "":
        splits = DATASETS[dataset_name]["splits"]
    return splits.split(",")


def get_meta(dataset_name):
    meta_json_path = os.path.join(const_names.METADATA_SUBDIR.format(dataset_name), const_names.METADATA_FINAL_FILE)
    return json.load(Path(meta_json_path).open("rt", encoding="utf8"))
