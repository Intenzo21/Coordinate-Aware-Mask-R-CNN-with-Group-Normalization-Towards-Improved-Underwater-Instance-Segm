"""
Script that runs the AMSRCR image enhancement technique on a given image dataset.
"""

import os
import cv2
import json
from glob import glob
from preprocessing.retinex.retinex import automatedMSRCR
from constants import TEST_IMG_DIR, IMG_DIR, AMSRCR_IMG_DIR, AMSRCR_TEST_IMG_DIR
from tqdm import tqdm
from utils import run_amsrcr_argument_parser

# Dictionary with dataset names and their associated
# original and target AMSRCR enhanced image directories.
DIR_DICT = {
    "train_val": [IMG_DIR, AMSRCR_IMG_DIR],
    "test": [TEST_IMG_DIR, AMSRCR_TEST_IMG_DIR]
}

if __name__ == "__main__":

    # Parse the command line arguments passed by the user.
    args = run_amsrcr_argument_parser().parse_args()

    # Get the original and target image directories
    # defined in constants.py.
    orig_dir, target_dir = DIR_DICT[args.dataset]

    # Change directories accordingly if provided by the user.
    if args.orig_dir:
        orig_dir = args.orig_dir
    if args.target_dir:
        target_dir = args.target_dir

    # Create the target directory.
    os.makedirs(target_dir, exist_ok=True)
    img_paths = glob(os.path.join(orig_dir, "*.jpg"))

    # Load the AMSRCR config file.
    config_path = os.path.join("preprocessing", "retinex", "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Perform AMSRCR image enhancement on the chosen dataset images
    # and save the enhanced images to the target directory.
    for img_p in tqdm(img_paths):

        img = cv2.imread(img_p)

        img_amsrcr = automatedMSRCR(img=img, sigma_list=config["sigma_list"])

        target_path = os.path.join(target_dir, os.path.basename(img_p))

        cv2.imwrite(target_path, img)
