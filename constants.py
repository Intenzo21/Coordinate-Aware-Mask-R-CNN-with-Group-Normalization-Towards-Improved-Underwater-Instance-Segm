"""
Script containing important constants.

Please change as needed.
"""
import os.path

# Seed value used for script seeding.
SEED = 42

# Name of the dataset configuration.
CONFIG_NAME = "aquatic"

# Helper dict with model name aliases.
FRAMEWORKS = {
    "mrcnn": "Mask R-CNN",
    "cmask": "CenterMask",
    "cinst": "CondInst",
    "solov2": "SOLOv2"
}

# Train and validation set folder, raw image,
# AMSRCR enhanced image and JSON directories.
TRAIN_VAL_FOLDER = os.path.join("data", "train_val")
IMG_DIR = os.path.join(TRAIN_VAL_FOLDER, "raw", "")
AMSRCR_IMG_DIR = os.path.join(TRAIN_VAL_FOLDER, "amsrcr", "")
JSON_DIR = os.path.join(TRAIN_VAL_FOLDER, "json", "")

# Test set folder, raw image,
# AMSRCR enhanced image and JSON directories.
TEST_FOLDER = os.path.join("data", "test")
TEST_IMG_DIR = os.path.join(TEST_FOLDER, "raw", "")
AMSRCR_TEST_IMG_DIR = os.path.join(TEST_FOLDER, "amsrcr", "")
TEST_JSON_DIR = os.path.join(TEST_FOLDER, "json", "")
