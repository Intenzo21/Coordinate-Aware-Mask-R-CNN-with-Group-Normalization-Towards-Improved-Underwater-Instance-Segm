"""
Script that implements the InstanceSegmentor class used for initialising
and training/evaluating instance segmentation models.
"""

import os
import json
import datetime
import timeit
from glob import glob

import matplotlib.image as mpimg

from detectron2.engine import DefaultPredictor
from detectron2 import config as d2_config

from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import ColorMode, Visualizer
from constants import SEED
from utils import register_datasets, adjust_dict
from custom_trainer import CustomTrainer


class InstanceSegmentor:
    """
    Create an end-to-end instance segmentation model.

    Currently, the only available options to choose from are Mask R-CNN (MRCNN),
    Coordinate-Aware Mask R-CNN (CAM-RCNN), CenterMask (CMask), CondInst (CInst) and SOLOv2.

    Examples:
    ::
        inst_segmentor = InstanceSegmentor()
        inst_segmentor.setup()
        inst_segmentor.train()
    """

    # Create a class constant since we use same base configuration,
    # pretrained weight and learning rate for both MRCNN and CAM-RCNN models.
    MRCNN_DICT = {
        "cfg_file": os.path.join("configs", "mrcnn", "mask_rcnn_R_101_FPN_3x.yaml"),
        "model_weights": os.path.join("coco_pretrained", "mrcnn", "model_final_a3ec72.pkl"),
        "lr": 0.001
    }

    # Create dictionary class constant with model names and their corresponding configurations.
    # This ensures neater code by avoiding the usage of multiple if-statements.
    MDL_DICT = {
        "mrcnn": MRCNN_DICT,
        "cam-rcnn": MRCNN_DICT,
        "cmask": {
            "cfg_file": os.path.join("configs", "cmask", "centermask_V_57_eSE_FPN_ms_3x.yaml"),
            "model_weights": os.path.join("coco_pretrained", "cmask", "centermask2-V-57-eSE-FPN-ms-3x.pth"),
            "lr": 0.0001
        },
        "cinst": {
            "cfg_file": os.path.join("configs", "cinst", "MS_R_101_3x.yaml"),
            "model_weights": os.path.join("coco_pretrained", "cinst", "CondInst_MS_R_101_3x.pth"),
            "lr": 0.001
        },
        "solov2": {
            "cfg_file": os.path.join("configs", "solov2", "R101_3x.yaml"),
            "model_weights": os.path.join("coco_pretrained", "solov2", "SOLOv2_R101_3x.pth"),
            "lr": 0.001
        }
    }

    def __init__(self, model="cam-rcnn"):
        """
        Initialise the chosen instance segmentation model.

        :param model: model name
        """

        self.model = model.lower()
        self.cfg_file = None  # To store the model config file
        self.model_weights = None  # To hold the model weights
        self.predictor = None  # To hold the model predictor instance
        self.cfg = None  # To store the model configuration
        self.epochs = None  # To store the model training epochs number

    def init_config(self, num_classes, aug):
        """
        Initialise the default model configuration.

        :param num_classes: number of classes in the dataset
        :param aug: a boolean flag to decide if augmentation will be employed
        :return: None
        """

        # Initialise the default configuration for the selected model.
        if "rcnn" in self.model:
            self.cfg = d2_config.get_cfg()
            self.cfg.set_new_allowed(True)
        elif self.model == "cmask":
            from centermask import config as cm_config
            self.cfg = cm_config.get_cfg()
        else:
            from adet import config as adet_config
            self.cfg = adet_config.get_cfg()
            if self.model == "solov2":

                # Set the SOLOv2 number of classes and confidence score.
                self.cfg.MODEL.SOLOV2.NUM_CLASSES = num_classes
                self.cfg.MODEL.SOLOV2.SCORE_THR = 0.5

            if not aug:  # Disable input flipping if not using augmentation.
                self.cfg.INPUT.HFLIP_TRAIN = False

    def setup(
            self,
            epochs=12,
            loss_type="dicebce",
            gn=True,
            aug=True,
            lrd=False,
            use_amsrcr=True,
            mbbnms=False,
            split_data=True,
            run_name="cam-rcnn"
    ):
        """
        Set up the chosen model for training/evaluation.

        :param epochs: number of training epochs
        :param loss_type: loss function to use for mask prediction
        :param gn: a boolean flag to determine whether to use Group Normalization
        layers or not (MRCNN models only)
        :param aug: a boolean flag to decide if augmentation will be employed
        :param lrd: a boolean flag to choose if learning rate decay will be utilised
        :param use_amsrcr: a boolean flag to decide if AMSRCR image enhancement will be utilised
        :param mbbnms: a boolean flag to choose if matrix bounding box non-maximum suppression
        technique will be utilised
        :param split_data: a boolean flag to choose if the dataset will be split into training (80%)
        and validation (20%) sets
        :param run_name: run folder name
        :return: None
        """

        # Quick check
        assert self.model in self.MDL_DICT.keys()

        # Set the number of epochs
        self.epochs = epochs

        # Get the number of iterations per epoch, number of classes and the configuration name
        epoch_iter, num_classes, cfg_name = register_datasets(use_amsrcr=use_amsrcr, split_data=split_data)

        # Initialise the base config file of the model
        if not self.cfg:
            self.init_config(num_classes, aug)

        # Get the configuration file, weights and base learning rate of the selected model.
        self.cfg_file, self.model_weights, lr = self.MDL_DICT[self.model].values()

        # Load model config and pretrained model.
        self.cfg.merge_from_file(self.cfg_file)
        self.cfg.MODEL.WEIGHTS = self.model_weights

        # Set the base learning rate, model minimum score threshold for the predicted instances
        # and the number of classes in the dataset.
        self.cfg.SOLVER.BASE_LR = lr  # 0.01 and 0.001 cause training issues (weights to explode).
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set testing (inference) threshold (default is 0.05)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        # Set the inference threshold and number of classes for the models
        # adopting FCOS, and the loss type, normalisation layer and
        # NMS type for CAM-RCNN
        if "rcnn" not in self.model:
            self.cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.5
            self.cfg.MODEL.FCOS.NUM_CLASSES = num_classes
        elif self.model == "cam-rcnn":
            self.cfg.MODEL.ROI_MASK_HEAD.LOSS = loss_type
            self.cfg.MODEL.ROI_MASK_HEAD.USE_COORD_CONV = True
            if gn:
                self.cfg.MODEL.ROI_MASK_HEAD.NORM = "GN"  # REMOVE WHEN INFERRING with baseline models
            if mbbnms:
                self.cfg.MODEL.ROI_HEADS.NMS_TYPE = "matrix"

        # Define the training and evaluation datasets to use for training
        self.cfg.DATASETS.TRAIN = (f"{cfg_name}_train",)

        # Whether to use a validation set or not
        if split_data:
            self.cfg.DATASETS.TEST = (f"{cfg_name}_val",)
        else:
            self.cfg.DATASETS.TEST = ()

        # Set the number of workers, images per batch, maximum number of iterations,
        # the warmup factor which is used in multiplying the learning rate
        # during the model warmup stage.
        self.cfg.DATALOADER.NUM_WORKERS = 1  # Since we want maximal reproducibility
        self.cfg.SOLVER.IMS_PER_BATCH = 1  # Since we want maximal reproducibility
        self.cfg.SOLVER.MAX_ITER = epoch_iter * self.epochs  # Maximum number of iterations (12 epochs = 4908)
        self.cfg.SOLVER.WARMUP_FACTOR = 0.001

        # Use half sized images and no random flip when not using augmentation.
        # This is typically employed when training baseline models.
        if not aug:
            self.cfg.INPUT.MIN_SIZE_TRAIN = (768,)  # 768 for .5 scaling
            self.cfg.INPUT.MIN_SIZE_TEST = 768  # 768 for .5 scaling
            self.cfg.INPUT.RANDOM_FLIP = "none"  # if is_train and cfg.INPUT.RANDOM_FLIP != "none":

        # Define the steps at which learning rate decay is performed.
        # We use the ratio seen in SOLOv2 proposal paper (https://arxiv.org/pdf/2003.10152.pdf).
        # That is, at 9th and 11th epochs when the total number of epochs is set to 12.
        if lrd:
            self.cfg.SOLVER.STEPS = \
                tuple(epoch_iter * self.epochs * r for r in (27 / 36, 33 / 36))
        else:
            self.cfg.SOLVER.STEPS = []  # do not decay learning rate  (NO DECAY BY DEFAULT STEPS = 210000 and 250000)

        # Seed the model configuration (data loader, augmentation, etc.)
        self.cfg.SEED = SEED

        # Create the output run folder using the current time and the
        # given model name
        now = datetime.datetime.now()
        segm_f = self.model if self.model == "cam-rcnn" else f"baseline_{self.model}"
        run_f = "{}_{:%Y%m%dT%H%M}".format(cfg_name, now)
        self.cfg.OUTPUT_DIR = os.path.join("output", segm_f, f"{run_f}_{run_name}")

    def train(self):
        """
        Train the selected instance segmentor model.

        :return: None
        """

        # Create the output directory to store the training output
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        # Save the model configuration as a YAML file
        with open(os.path.join(self.cfg.OUTPUT_DIR, "cfg.yaml"), "w") as f:
            f.write(self.cfg.dump())

        # Initialise the CustomTrainer instance
        trainer = CustomTrainer(self.cfg, self.epochs)

        # Start from the 0 epoch
        trainer.resume_or_load(resume=False)

        # Train for the specified number of epochs
        trainer.train()

    def get_prediction(self, image_paths, model_path, test_set, use_amsrcr):
        """
        Get the instance prediction image of the provided model.


        :param image_paths: paths to the images to make predictions on
        :param model_path: path to the pretrained model
        :param test_set: a boolean flag to determine if the test set will be used
        :param use_amsrcr: a boolean flag to determine if AMSRCR image enhancement will be used
        :return: image with the predicted instances
        """

        # Load the pretrained model weights
        self.cfg.MODEL.WEIGHTS = model_path

        # Register the test set.
        if test_set:
            try:
                self.cfg.DATASETS.TEST = (register_datasets(test_set=True, use_amsrcr=use_amsrcr),)
            except AssertionError:
                pass

        # Initialise the model predictor
        self.predictor = DefaultPredictor(cfg=self.cfg)

        # Loop over the image paths provided
        for img_p in glob(image_paths):

            # Read the image from the provided path and
            # convert it to BGR format since DefaultPredictor
            # expects BGR-formatted input
            img = mpimg.imread(img_p)[:, :, ::-1]

            # Perform prediction on the image
            prediction = self.predictor(img)

            # Create a visualiser instance
            viz = Visualizer(img, MetadataCatalog.get(self.cfg.DATASETS.TEST[0]))  # instance_mode = ColorMode.BW

            # Visualise the prediction
            out = viz.draw_instance_predictions(prediction["instances"].to("cpu"))

            # Convert back to RGB for visualisation and yield
            yield out.get_image()[:, :, ::-1]

    def evaluate(
            self,
            model_path,
            test_set=False,
            tta=False,
            inference_folder=None,
            use_amsrcr=True
    ):
        """
        Run model evaluation.

        :param model_path: path to the pretrained model
        :param test_set: a boolean flag to choose if the evaluation will be performed on the test or validation set
        :param tta: a boolean flag to choose if test-time augmentation will be adopted
        :param inference_folder: name of the inference folder to save the results
        :param use_amsrcr: a boolean flag to determine if the AMSRCR image enhancement technique will be used
        :return: None
        """
        self.cfg.MODEL.WEIGHTS = model_path

        # Register the test set.
        if test_set:
            try:
                self.cfg.DATASETS.TEST = (register_datasets(test_set=True, use_amsrcr=use_amsrcr),)
            except AssertionError:
                pass

        # Get the evaluation dataset name.
        dataset_name = self.cfg.DATASETS.TEST[0]

        # Initialise the model predictor
        self.predictor = DefaultPredictor(cfg=self.cfg)

        # Specify the output directory.
        output_dir = os.path.dirname(model_path)

        # Define a default inference folder name if not provided.
        if not inference_folder:
            inference_folder = f"{dataset_name.split('_')[1]}_inference"

        # Initialise the CustomTrainer test function with TTA if needed
        # DefaultTrainer utilises PIL so RGB images are converted to BGR
        # internally for predictions (inference)
        if tta:
            test_func = CustomTrainer.test_with_tta
            inference_folder += "_TTA"
        else:
            test_func = CustomTrainer.test

        # Define the output folder.
        output_folder = os.path.join(output_dir, inference_folder)

        # Instantiate a COCOEvaluator for evaluation.
        evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)

        # Perform evaluation and get the metrics.
        metrics = test_func(self.cfg, self.predictor.model, evaluators=[evaluator])

        # Uncomment for evaluating the inference time
        # from statistics import mean
        # t = timeit.repeat(lambda: test_func(self.cfg, self.predictor.model, evaluators=[evaluator]),
        #                   number=1, repeat=3)
        # txt_path = os.path.join(output_folder, "timeit.txt")
        # with open(txt_path, 'w') as f:
        #     f.write(json.dumps(mean(t)))

        # Adjust the metrics dictionary and save it to a text file.
        metrics = adjust_dict(metrics)
        txt_path = os.path.join(output_folder, "eval_metrics.txt")
        with open(txt_path, 'w') as f:
            f.write(json.dumps(metrics))
