"""
Script that implements the validation loss evaluation hook for Detectron2.

Adapted from: https://eidos-ai.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
"""

import logging
import numpy as np
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds, setup_logger
import detectron2.utils.comm as comm
import torch
import time
import datetime


class LossEvalHook(HookBase):
    """
    Create a custom validation loss evaluation hook.
    """

    def __init__(self, eval_period, model, data_loader):
        """
        Initialise the Detectron2 custom evaluation hook.

        :param eval_period: period of performing validation loss evaluations (in iterations)
        :param model: Detectron2 model
        :param data_loader: Detectron2 data loader instance
        """

        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        setup_logger(name=__name__)

    def _do_loss_eval(self):
        """
        Perform loss evaluation.

        :return: averaged loss over the batch
        """
        # Copying inference_on_dataset from evaluator.py.
        total = len(self._data_loader)

        # Define the warmup iterations number.
        num_warmup = min(5, total - 1)

        # Record the starting time.
        start_time = time.perf_counter()

        # Total time for the loss calculation.
        total_compute_time = 0

        # Iterate over the input batches, calculate the loss for each batch,
        # sum the batch losses and append the summed loss to the list of losses
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on validation set: done {}/{} --- {:.4f} s / img --- ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)

        # Calculate the average of the list of batch losses
        mean_loss = np.mean(losses)

        # Log the validation loss in the terminal
        self.trainer.storage.put_scalar('val_loss', mean_loss)
        comm.synchronize()

        return mean_loss  # losses

    def _get_loss(self, data):
        """
        Calculate the overall loss of the batch of inputs.

        :param data: batch of inputs
        :return: summed (reduced) loss of the batch of inputs
        """
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        """
        Perform the validation loss evaluation after the training step at
        every evaluation period.

        :return: None
        """

        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()

