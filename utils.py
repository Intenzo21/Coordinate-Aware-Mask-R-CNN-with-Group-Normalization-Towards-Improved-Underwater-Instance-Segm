"""
Script that implements various utility functions.
"""

import argparse
import os
import random
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
import json
from glob import glob

from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from constants import *

import matplotlib.pyplot as plt

from collections import defaultdict
from textwrap import wrap


def reset_seeds(seed=SEED):
    """
    Reset the random number generator seed to achieve reproducibility.

    :param seed: seed value
    :return: None
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger = setup_logger(name=__name__)
    logger.info("All seeds reset!")


def get_classes(json_paths):
    """
    Get the class names in the dataset.

    :param json_paths: annotation JSON file paths
    :return: list of class names in the dataset
    """

    # List to store the class names
    cls_names = []

    # Load the annotation JSON files and append unique class names
    # to the class list
    for j in json_paths:
        with open(j) as jf:
            annot = json.load(jf)
        img_shapes = annot["shapes"]
        for sh in img_shapes:
            sh_label = sh["label"]
            if sh_label not in cls_names:
                cls_names.append(sh_label)

    return cls_names


def get_aquatic_dicts(json_paths, img_dir, cls_names):
    """
    Get the aquatic dataset dictionaries containing dataset image file and annotations info
    such as image height and width, path to each image file, instance label (class), etc.

    :param json_paths: annotation JSON file paths
    :param img_dir: aquatic dataset image directory
    :param cls_names: dataset class names
    :return: aquatic dataset dictionary
    """

    # List to store the dataset dictionary records
    dataset_dicts = []

    # Iterate over the JSON files
    for idx, j_path in enumerate(json_paths):

        # Load annotations
        with open(j_path) as jf:
            annot = json.load(jf)

        # Get the current filename
        filename = os.path.join(img_dir, annot["imagePath"])

        # Get the current image height and width
        h, w = annot["imageHeight"], annot["imageWidth"]

        # Store above info in a record
        record = {
            "file_name": filename,
            "image_id": idx,
            "height": h,
            "width": w,
        }

        # Get the annotation shapes
        img_shapes = annot["shapes"]

        # Create a list to store the objects
        objs = []

        # Iterate over the image shapes
        for sh in img_shapes:
            # Get the shape points and label (class)
            sh_points = sh["points"]
            sh_label = sh["label"]

            px, py, poly = [], [], []

            # Format the x-axis and y-axis points as
            # required by Detectron2
            for x, y in sh_points:
                px.append(x)
                py.append(y)
                poly.extend([x, y])

            # Store bounding box and segmentation info of the object (instance)
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": cls_names.index(sh_label),
            }

            # Append the object to the list of objects
            objs.append(obj)

        # Store the objects info under the "annotations" key in the record
        record["annotations"] = objs

        # Append the record to the dataset dictionaries (records) list
        dataset_dicts.append(record)

    return dataset_dicts


def register_datasets(test_set=False, use_amsrcr=False, split_data=True):
    """
    Register custom datasets to Detectron2.

    :param test_set:
    :param use_amsrcr:
    :param split_data:
    :return: number of iterations in an epoch, number of classes and configuration name ("aquatic")
    """

    # Get the config name from the constants file
    cfg_name = CONFIG_NAME.lower()

    # Get the JSON paths and the dataset class names
    train_val_jps = tuple(glob(f"{JSON_DIR}*.json"))
    cls_names = get_classes(train_val_jps)

    # Register the test set
    if test_set:
        test_jps = tuple(glob(f"{TEST_JSON_DIR}*.json"))
        ds_name = f"{cfg_name}_test"
        img_dir = AMSRCR_TEST_IMG_DIR if use_amsrcr else TEST_IMG_DIR
        # cls_names = get_classes(jp)
        DatasetCatalog.register(ds_name, lambda jps=test_jps: get_aquatic_dicts(jps, img_dir, cls_names))
        MetadataCatalog.get(ds_name).set(thing_classes=cls_names)
        return ds_name

    # Split the data into training and validation sets
    if split_data:
        datasets = dict(
            zip(
                ["train", "val"],
                train_test_split(train_val_jps, test_size=0.2, random_state=SEED)
            )
        )
    else:
        datasets = {"train": train_val_jps}

    # Get the iteration numbers in an epoch, number of classes
    # and the dataset image directory
    epoch_iters = int(len(datasets["train"]))
    num_classes = len(cls_names)
    img_dir = AMSRCR_IMG_DIR if use_amsrcr else IMG_DIR

    # Register the training and validation sets to Detectron2
    try:
        for ds, j_paths in datasets.items():
            DatasetCatalog.register(f"{cfg_name}_" + ds, lambda jps=j_paths: get_aquatic_dicts(jps, img_dir, cls_names))
            MetadataCatalog.get(f"{cfg_name}_" + ds).set(thing_classes=cls_names)
    except AssertionError:
        pass

    return epoch_iters, num_classes, cfg_name


def get_json_lines(json_path):
    """
    Get each line of a JSON file.

    :param json_path: path to the JSON file
    :return: the fetched JSON file lines
    """
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def save_val_results(run_folders, same_fw=False):
    """
    Save instance segmentation metric results using the JSON file
    generated at the end of the model training. Typically, we adopt the validation set
    during training.

    :param run_folders: folders containing the metrics JSON file
    :param same_fw: a boolean flag to determine if the results are related to a single or multiple models
    :return: None
    """
    # Get all run output paths from the regex
    run_paths = glob(run_folders)

    # List to store the metric dictionaries
    metrics_dicts = []

    # Loop over the metrics folders
    for rp in run_paths:

        # Get the run name from the metrics folder name
        rp_split = rp.split(os.sep)
        fw_f, run_f_name = rp_split[-2], rp_split[-1]

        if same_fw:
            fw_name = fw_f
        else:
            fw_name = FRAMEWORKS[fw_f.split("_")[1]]

        run_name = f"{fw_name} ({'_'.join([rn for rn in run_f_name.split('_')[2:]])})"

        # Get the metrics from the JSON file lines (last line)
        try:
            experiment_metrics = get_json_lines(rp + '/metrics.json')[-1]
        except FileNotFoundError:
            print("File not found!")
            continue

        # Insert the run name into the metrics dict
        experiment_metrics["Framework"] = run_name

        # Append the current metrics dictionary to the list dictionaries
        metrics_dicts.append(experiment_metrics)

    # Save the metrics result for each run and the averaged results into separate CSV files
    metrics_res = pd.DataFrame(metrics_dicts)
    metrics_res_mean = metrics_res.groupby("Framework").mean()

    metrics_res.set_index("Framework").sort_index().to_csv("val_results.csv")
    metrics_res_mean.to_csv("val_results_mean.csv")


def save_infer_results(run_folders, metrics_folders=None, same_fw=True, csv_name=None):
    """
    Save inference results.

    :param run_folders: run output folders
    :param metrics_folders: folders containing the metrics dictionary text file
    :param same_fw: a boolean flag to determine if the results are related to a single or multiple models
    :param csv_name: name of the CSV file to save the results
    :return: None
    """

    # Get all run output paths from the regex
    run_paths = glob(run_folders)

    # Specify a default metrics folder if not provided
    if not metrics_folders:
        metrics_folders = ["test_inference"]

    # Define a default CSV file name if not provided
    if not csv_name:
        csv_name = "test"

    # Create an empty list to store the metrics dictionaries
    metrics_dicts = []

    # Loop over the output folders
    for rp in run_paths:

        # Get all metrics folder paths in the current output folder using the regex
        mps = glob(os.path.join(rp, metrics_folders))

        # Loop over the metrics folder paths
        for mp in mps:

            # Get the current metrics text file path
            m_txt_path = os.path.join(mp, "eval_metrics.txt")

            # Read the metrics text file
            try:
                with open(m_txt_path) as f:
                    metrics = f.read()
            except FileNotFoundError:
                print(f"File '{m_txt_path}' not found!")
                continue

            # Get the run name from the output folder name
            rp_split = rp.split(os.sep)
            fw_f, run_f_name = rp_split[-2], rp_split[-1]

            if same_fw:
                fw_name = fw_f
            else:
                fw_name = FRAMEWORKS[fw_f.split("_")[1]]

            run_name = f"{fw_name} ({'_'.join([rn for rn in run_f_name.split('_')[2:]])}/{os.path.basename(mp)})"

            # Load the metrics dictionary
            metrics_dict = json.loads(metrics)

            # Specify the bounding box and segmentation metrics
            # keys as defined in Detectron2
            bb_key = "bbox"
            segm_key = "segm"

            # Adjust if using TTA
            if "TTA" in mp:
                run_name += " + TTA"
                bb_key += "_TTA"
                segm_key += "_TTA"

            # Adjust the metrics dictionary if needed
            metrics_dict = adjust_dict(metrics_dict, tta="TTA" in mp)

            if "val" in mp:
                metrics_dict = {k: metrics_dict[k] if k != "iteration" else 4908 for k in metrics_dict.keys()}

            # Insert the run name into the metrics dict
            flattened_dict = {**metrics_dict, "Framework": run_name}

            # Append the current metrics dictionary to the list dictionaries
            metrics_dicts.append(flattened_dict)

    # Save the metrics result for each run and the averaged results into separate CSV files
    metrics_res = pd.DataFrame(metrics_dicts)
    metrics_res_mean = metrics_res.groupby("Framework").mean()

    metrics_res.set_index("Framework").sort_index().to_csv(f"{csv_name}_results.csv")
    metrics_res_mean.to_csv(f"{csv_name}_results_mean.csv")


def adjust_dict(metrics_dict, tta=False):
    """
    Adjust a metrics dictionary for ease of use.

    :param metrics_dict: the metrics dictionary
    :param tta: a boolean flag to choose if test-time augmentation will be utilised
    :return: adjusted metrics dictionary (w/ correct keys for saving and plotting purposes)
    """

    # Define the bounding box and segmentation keys.
    bb_key = "bbox"
    segm_key = "segm"

    # Adjust if using TTA.
    if tta:
        bb_key += "_TTA"
        segm_key += "_TTA"

    # Try adjusting the metrics dictionary and if already adjusted return as is.
    try:
        bbox_metrics = {f"bbox/{key}": val for key, val in metrics_dict[bb_key].items()}
        segm_metrics = {f"segm/{key}": val for key, val in metrics_dict[segm_key].items()}
    except KeyError:
        return metrics_dict

    flattened_dict = {**bbox_metrics, **segm_metrics}
    return flattened_dict


def show_annots():
    """
    Show image annotations.

    Used for debugging purposes.

    :return: None
    """
    json_dir = "data/test/json/"
    img_dir = "data/test/raw/"
    ds_name = "aquatic_test"

    ds_paths = glob(f"{json_dir}*.json")
    cls_names = get_classes(ds_paths)
    ds_dicts = get_aquatic_dicts(ds_paths, img_dir, cls_names)

    DatasetCatalog.register(ds_name, lambda jp=tuple(ds_paths): get_aquatic_dicts(jp, img_dir, cls_names))
    MetadataCatalog.get(ds_name).set(thing_classes=cls_names)
    test_metadata = MetadataCatalog.get(ds_name)

    for d in ds_dicts[:3]:
        img = mpimg.imread(d["file_name"])
        visualizer = Visualizer(img, metadata=test_metadata)
        out = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(12, 12))
        plt.imshow(out.get_image())
        plt.show()


def count_instances(json_paths):
    """
    Count number of instances in the dataset.

    :param json_paths: annotation JSON file paths
    :return: number of instances in the dataset
    """

    count = 0

    # Load the annotation JSON files and count the shapes.
    for jp in json_paths:
        with open(jp) as jf:
            annot = json.load(jf)
        count += len(annot["shapes"])
    return count


# import memory_profiler
# @memory_profiler.profile(precision=5)
def plot_losses(run_folders, model_name="Model"):
    """
    Plot the training and/or validation loss of a model.

    :param run_folders: list of folders containing the model metrics file (metrics.json)
    :param model_name: name of the model
    :return: None
    """
    run_folders = glob(run_folders)
    # Number of runs used for averaging the losses
    num_runs = len(run_folders)
    total_losses, val_losses = defaultdict(float), defaultdict(float)

    # Loop over the metrics folders
    for rf in run_folders:

        metrics_path = os.path.join(rf, "metrics.json")
        assert os.path.isfile(metrics_path)

        # Get the metrics JSON file lines
        experiment_metrics = get_json_lines(metrics_path)

        # Average the total and validation losses over the number of runs performed
        for x in experiment_metrics:
            if "total_loss" in x:
                total_losses[x["iteration"]] += x["total_loss"] / num_runs
            if "val_loss" in x:
                val_losses[x["iteration"]] += x["val_loss"] / num_runs

    # plt.figure(figsize=(8, 6))
    # Plot the averaged total and validation losses
    for losses in [total_losses, val_losses]:
        plt.plot(
            losses.keys(),
            losses.values()
        )

    plt.title("\n".join(wrap(f"{model_name.capitalize()} Average Training Loss", 50)))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(['Training loss'], loc='upper right')
    plt.tight_layout()
    plt.show()


def show_plot(ax, total_counts, total_annot=True, title="", rotation=30, use_legend=True):
    """
    Setup and show plot of the passed ax.

    :param ax: ax to plot (from pandas DataFrame)
    :param total_counts: total distribution counts
    :param total_annot: a boolean flag to determine whether to plot
    the total counts or not
    :param title: the title of the plot
    :param rotation: rotation for the x-axis ticks
    :param use_legend: a boolean flag to determine whether to include a legend
    or not
    :return: None
    """

    # Iterate over the ax patches and annotate accordingly
    for i, p in enumerate(ax.patches):

        # Get the bar coordinates, width and height
        x, y = p.get_xy()
        x += p.get_width() / 2
        y += p.get_height() / 2

        # Center bar values if above 5
        if p.get_height() >= 5:
            ax.annotate(str(int(p.get_height())), (x, y), ha='center', va='center', c='white', fontsize=11)

        # Top bar values
        if ((i + 1 <= len(ax.patches) / 3) or len(ax.patches) == 3) and total_annot:
            ax.annotate(str(int(total_counts[i])), (x, int(total_counts[i]) + 2), ha='center', c='black', fontsize=11)

    # Setup the plot and show
    plt.xticks(rotation=rotation, horizontalalignment="center", fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(title)
    plt.ylabel("Count", fontdict={"size": 11})
    if use_legend:
        plt.legend(fontsize=11)
    elif ax.get_legend():
        ax.get_legend().remove()
    plt.tight_layout()
    plt.show()


def plot_distributions(train_val_paths, test_paths):
    """
    Plot the instance per category and image distributions over
    the training, validation and test sets.

    :param train_val_paths: training and validation set JSON file paths
    :param test_paths: test set JSON file paths
    :return: None
    """

    inst_dict = {
        "Training": defaultdict(int),
        "Validation": defaultdict(int),
        "Test": defaultdict(int),
    }

    # Split to train and validation sets
    datasets = dict(
        zip(
            list(inst_dict.keys())[:-1],
            train_test_split(train_val_paths, test_size=0.2, random_state=SEED)
        )
    )

    # Insert the test set paths to the datasets dictionary
    datasets["Test"] = test_paths

    # To store the total image counts for each set
    img_counts = {}

    # Iterate over the datasets dictionary
    # and count the images and instances per category
    for k, v in datasets.items():
        img_counts[k] = len(v)
        for jp in v:
            with open(jp) as jf:
                annot = json.load(jf)

            img_shapes = annot["shapes"]

            for sh in img_shapes:
                inst_dict[k][sh["label"]] += 1

    # Create a Pandas data frame from the instance count dictionary
    inst_df = pd.DataFrame(inst_dict)
    # print(inst_df)
    # print(inst_df.sum(axis=0))

    # Plot the instance data frame as a bar chart
    inst_ax = inst_df.plot(kind="bar", stacked=True)
    show_plot(
        inst_ax,
        total_counts=inst_df.sum(axis=1),
        title="Training, Validation and Test Set Instances Per Category"
    )

    # Create a Pandas data frame from the instance count dictionary
    img_df = pd.DataFrame(img_counts, index=["Count"])

    # Plot the image data frame as a bar chart
    img_ax = img_df.T.plot(kind="bar", color="red")
    show_plot(
        img_ax,
        total_counts=img_df.T["Count"],
        total_annot=False,
        title="Training, Validation and Test Set Image Distribution",
        use_legend=False,
        rotation=0
    )

    # Plot the total instances per set as a bar chart
    tinst_ax = inst_df.sum(axis=0).plot(kind="bar", color="purple")
    show_plot(
        tinst_ax,
        total_counts=inst_df.sum(axis=0),
        total_annot=False,
        title="Training, Validation and Test Set Total Instances Distribution",
        use_legend=False,
        rotation=0
    )


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """
    Custom formatter class that fuses argparse.ArgumentDefaultsHelpFormatter
    and argparse.RawDescriptionHelpFormatter to get the best of both worlds.
    For further info, please refer to: https://docs.python.org/3/library/argparse.html#formatter-class
    """
    pass


def inst_segm_argument_parser(epilog=None):
    """
    Create the command line argument parser for the instance segmentor
    training script (train_eval_model.py).

    :param epilog: text to display after the argument help
    :return: argument parser
    """

    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
    Examples:

    Train best performing CAM-RCNN model (with DBL, AUG, AMSRCR and TTA)
    for 3 runs of 12 epochs (by default), as presented in our paper:
        $ python {sys.argv[0]} cam-rcnn 

    Run evaluation on the test set without TTA using all baseline pretrained CAM-RCNN models.
    Results are saved to a folder titled "test_inference" by default:
        $ python {sys.argv[0]} cam-rcnn --eval-only output/cam-rcnn/*_baseline
    
    Run prediction on AMSRCR enhanced test set images with the best performing CAM-RCNN pretrained model:
        $ python {sys.argv[0]} cam-rcnn --predict data/test/amsrcr/*.jpg model_final.pth 
    """,
        formatter_class=CustomFormatter,
    )

    def check_positive(val):
        """
        Check if the value provided is positive.

        :param val: the value to check
        :return: the value if it passes the check
        """

        val = int(val)
        if val <= 0:
            raise argparse.ArgumentTypeError(f"Expected a positive integer got: {val}")
        return val

    # Add an argument to specify the model name.
    parser.add_argument(
        "model",
        type=str,
        choices=["mrcnn", "cam-rcnn", "cmask", "cinst", "solov2"],
        metavar="MODEL_NAME",
        help=f"model name from: [%(choices)s]"
    )

    # Add an argument to disable script seeding.
    parser.add_argument(
        "--unseeded",
        action="store_true",
        help="disable script seeding"
    )

    # Add an argument to specify the number of training epochs.
    parser.add_argument(
        "--epochs",
        default=12,
        type=check_positive,
        metavar="EPOCHS",
        help="number of training epochs"
    )

    # Add an argument to specify the loss function to use for the CAM-RCNN model.
    parser.add_argument(
        "--loss",
        default="dicebce",
        choices=["bce", "dice", "dicebce", "tversky", "explog", "logcosh"],
        metavar="LOSS_NAME",
        help="instance segmentation loss function to use (applies only for the CAM-RCNN model): [%(choices)s]"
    )

    # Add an argument to disable group normalisation.
    parser.add_argument(
        "--no-gn",
        action="store_true",
        help="do not use Group Normalization layers (applies only for the CAM-RCNN model)"
    )

    # Add an argument to disable training-time augmentation.
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="disable image resizing and flipping (augmentation) during training"
    )

    # Add an argument to specify that original images will be used for training and inference.
    parser.add_argument(
        "--no-amsrcr",
        action="store_true",
        help="use original images during training and inference"
    )

    # Add an argument to enable learning rate decay.
    parser.add_argument(
        "--lrd",
        action="store_true",
        help="use learning rate decay"
    )

    # Add an argument to enable MBBNMS.
    parser.add_argument(
        "--mbbnms",
        action="store_true",
        help="use matrix bounding box non-maximum suppression"
    )

    # Add an argument to specify the run name.
    parser.add_argument(
        "--run-name",
        default="baseline",
        metavar="RUN_NAME",
        help="name of the current run folder (where the pretrained model will be saved)"
    )

    # Add an argument to specify the number of runs.
    parser.add_argument(
        "--run-count",
        default=3,
        type=check_positive,
        metavar="COUNT",
        help="how many times to train the model"
    )

    # Add an argument to specify that the user wants to evaluate a pretrained model.
    parser.add_argument(
        "--eval-only",
        default=False,
        metavar="MODEL_PATHS",
        help="evaluate pretrained models from the given paths (use regex for "
             "multiple folders)"
    )

    # Add an argument to define parameters used in model evaluation.
    parser.add_argument(
        "--eval-args",
        default=["True", "False", "test_inference"],
        nargs=3,
        metavar=("TEST_SET", "TTA", "INF_FOLDER_NAME"),
        help="booleans to enable/disable test set and TTA usages, and a name"
             " for the inference folder to store the evaluation results"
    )

    # Add an argument to enable making predictions using the provided image paths and
    # model pretrained weights.
    parser.add_argument(
        "--predict",
        default=False,
        nargs=2,
        metavar=("IMAGE_PATHS", "MODEL_PATH"),
        help="make prediction on given test/validation set images using the "
             " provided pretrained weights"
    )

    # Add an argument to define parameters used in model evaluation.
    parser.add_argument(
        "--pred-set",
        choices=["test", "val"],
        default="test",
        metavar="PRED_SET",
        help="test/validation set to use for predictions"
    )

    return parser


def inspect_results_argument_parser(epilog=None):
    """
    Create the command line argument parser for inspect_results.py.

    :param epilog: text to display after the argument help
    :return: argument parser
    """

    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
    Examples:
    
    Plot baseline (default) CAM-RCNN model averaged training and validation losses.
    Please define the metrics folders using the regex "*" symbol to get all folders 
    automatically. The command line command below utilises all folders ending with "_baseline":
        $ python {sys.argv[0]} --plot-losses ./output/cam-rcnn/*_baseline CAM-RCNN
        
    Save baseline (default) CAM-RCNN model metrics results using the output text files.
    Please define the metrics folders using the regex "*" symbol to get all folders 
    automatically. The command line command below utilises all output folders ending with "_baseline"
    and all metrics folders starting with "test_". Alternatively, one can specify only a single metric 
    folder such as "test_inference":
        $ python {sys.argv[0]} --save-txt ./output/cam-rcnn/*_baseline test_* True "test_folder_results"  
    """,
        formatter_class=CustomFormatter,
    )

    # Add an argument to define model run folders and model name (used for the title).
    parser.add_argument(
        "--plot-losses",
        default=False,
        nargs=2,
        metavar=("RUN_FOLDERS", "MODEL_NAME"),
        help="plot losses of the instance segmentation model"
    )

    # Add an argument used for obtaining metrics summary as a CSV file
    # from multiple model metrics text files.
    parser.add_argument(
        "--save-txt",
        default=False,
        nargs=4,
        metavar=("RUN_FOLDERS", "METRICS_FOLDERS", "SAME_FW", "CSV_NAME"),
        help="save model metric results from the outputted text file"
    )

    # Add an argument used for obtaining metrics summary as a CSV file
    # from multiple model metrics JSON files.
    parser.add_argument(
        "--save-json",
        default=False,
        nargs=2,
        metavar=("RUN_FOLDERS", "MODEL_NAME"),
        help="plot model metric results from the outputted JSON file"
    )

    return parser


def run_amsrcr_argument_parser(epilog=None):
    """
    Create the command line argument parser for run_amsrcr.py.

    :param epilog: text to display after the argument help
    :return: argument parser
    """

    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
    Examples:

    Perform AMSRCR image enhancement on the aquatic test set:
        $ python {sys.argv[0]} test
    """,
        formatter_class=CustomFormatter,
    )

    # Add an argument to specify which image dataset to use for AMSRCR image enhancement.
    parser.add_argument(
        "dataset",
        choices=["train_val", "test"],
        metavar="DATASET_NAME",
        help=f"dataset for the AMSRCR image enhancement: [%(choices)s]"
    )

    # Add an argument to change the original image directory.
    parser.add_argument(
        "--orig-dir",
        default=False,
        type=str,
        metavar="ORIG_DIR",
        help=f"original image directory"
    )

    # Add an argument to change the target image directory.
    parser.add_argument(
        "--target-dir",
        default=False,
        type=str,
        metavar="TARGET_DIR",
        help=f"target directory to store the AMSRCR enhanced images"
    )

    return parser


def plot_dist_argument_parser(epilog=None):
    """
    Create the command line argument parser for plot_dist.py.

    :param epilog: text to display after the argument help
    :return: argument parser
    """

    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
    Examples:
    
    Plot the training, validation and test set instances per category and image distributions.
    Please use regex (i.e. "*" as below) to specify that all JSON files are to be used:
        $ python {sys.argv[0]} data/train_val/json/*.json data/test/json/*.json
    
    Note that the plots will be displayed one by one!
    """,
        formatter_class=CustomFormatter,
    )

    # Add an argument to specify the path to the training and validation set JSON file paths.
    parser.add_argument(
        "train_val_paths",
        default=False,
        help=f"training and validation set JSON file paths"
    )

    # Add an argument to specify the path to the test set JSON file paths.
    parser.add_argument(
        "test_paths",
        default=False,
        help=f"test set JSON file paths"
    )

    return parser
