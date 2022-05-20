"""
Script that implements the CustomTrainer class.
"""
import logging
import os
from collections import OrderedDict

from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.modeling import GeneralizedRCNNWithTTA

from loss_eval_hook import LossEvalHook


class CustomTrainer(DefaultTrainer):
    """
    Subclass Detectron2's DefaultTrainer.

    Introduce the "epochs" instance variable, override the "build_evaluator" method,
    plug in the LossEvalHook into the training process and define a new model testing method
    that adopts TTA.
    """

    def __init__(self, cfg, epochs):
        """
        Initialise the CustomTrainer class instance.

        :param cfg: model configuration
        :param epochs: number of training epochs
        """

        self.epochs = epochs
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Build a COCO-style dataset evaluator.

        :param cfg: model configuration
        :param dataset_name: name of the dataset
        :param output_folder: folder to save the inference results
        :return: COCO-style data evaluator (COCOEvaluator)
        """

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        """
        Build the hooks of the DefaultTrainer and add the custom
        LossEvalHook.

        :return: DefaultTrainer hooks with the added LossEvalHook
        """

        hooks = super().build_hooks()
        if self.cfg.DATASETS.TEST:
            hooks.insert(
                -1,
                LossEvalHook(
                    self.cfg.SOLVER.MAX_ITER / self.epochs,
                    self.model,
                    build_detection_test_loader(
                        self.cfg,
                        self.cfg.DATASETS.TEST[0],
                        DatasetMapper(self.cfg, True)
                    )
                )
            )
        return hooks

    @classmethod
    def test_with_tta(cls, cfg, model, evaluators):
        """
        Perform inference with TTA.

        Only support some R-CNN models.

        :param cfg: model configuration
        :param model: instance segmentation model
        :param evaluators: list of evaluators to use
        :return: TTA inference results dictionary
        """

        # Setup the logger
        setup_logger(name=__name__)
        logger = logging.getLogger(__name__)
        logger.info("Running inference with test-time augmentation ...")

        # Wrap the model so that it can adopt TTA
        model = GeneralizedRCNNWithTTA(cfg, model)

        # Perform the evaluation and save the results
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})

        return res
