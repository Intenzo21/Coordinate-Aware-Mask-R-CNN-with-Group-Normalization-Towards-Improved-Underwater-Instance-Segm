#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Script used for visualising model JSON IS results (i.e. coco_instances_results.json).

Adapted from: https://github.com/facebookresearch/detectron2/blob/main/tools/visualize_json_results.py
"""

import argparse
import json
import random

import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode

from utils import register_datasets, CustomFormatter
from matplotlib import colors


def create_instances(predictions, image_size):
    """
    Create Instances object from predictions.

    :param predictions: inference predictions
    :param image_size: image size
    :return: the constructed Instances object with scores, boxes, labels, etc.
    """
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([predictions[i]["category_id"] for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":

    # Create the argument parser and parse the command line arguments
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset.",
        formatter_class=CustomFormatter
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--no-amsrcr", help="disable AMSRCR image enhancement", action="store_true")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    # Initialise a logger
    logger = setup_logger()

    # Load the JSON file
    with PathManager.open(args.input, "r") as f:
        preds = json.load(f)

    # Append predictions according to each image id
    pred_by_image = defaultdict(list)
    for p in preds:
        pred_by_image[p["image_id"]].append(p)

    # Register the necessary dataset
    register_datasets(test_set="test" in args.dataset, use_amsrcr=not args.no_amsrcr)

    # Get the dataset dictionaries and metadata
    # containing info about each image
    dicts = list(DatasetCatalog.get(args.dataset))

    # Get reproducible colors for instances
    clrs = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
    random.seed(42)
    random.shuffle(clrs)
    metadata = MetadataCatalog.get(args.dataset)
    clrs = clrs[:len(metadata.thing_classes)]
    metadata.set(thing_colors=[colors.to_rgb(c=clr) for clr in clrs])

    # Create the ground-truth images folder
    gt_dir = os.path.join(args.output, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    # Create the prediction images folder
    pred_dir = os.path.join(args.output, "pred")
    os.makedirs(pred_dir, exist_ok=True)

    # Iterate over the dataset dictionaries and save the ground-truth images
    # and instance prediction images to separate folders
    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        preds = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata, instance_mode=ColorMode.SEGMENTATION, fontsize=36)
        vis_pred = vis.draw_instance_predictions(preds).get_image()
        cv2.imwrite(os.path.join(pred_dir, basename), vis_pred[:, :, ::-1])

        vis = Visualizer(img, metadata, instance_mode=ColorMode.SEGMENTATION, fontsize=36)
        vis_gt = vis.draw_dataset_dict(dic).get_image()
        cv2.imwrite(os.path.join(gt_dir, basename), vis_gt[:, :, ::-1])

