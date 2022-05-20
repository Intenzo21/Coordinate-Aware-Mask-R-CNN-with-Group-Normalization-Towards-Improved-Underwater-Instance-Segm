"""
Script used for training an instance segmentation model.
"""

import os
from glob import glob

import matplotlib.pyplot as plt

from instance_segmentor import InstanceSegmentor

from utils import inst_segm_argument_parser, reset_seeds
from distutils.util import strtobool
from constants import *

if __name__ == "__main__":

    # Parse the arguments passed to the terminal.
    args = inst_segm_argument_parser().parse_args()

    # Loop for the number of runs specified
    for _ in range(args.run_count):

        # Seed the training script if needed.
        if not args.unseeded:
            reset_seeds()

        # Create the instance segmentation model.
        inst_segmentor = InstanceSegmentor(model=args.model)

        # Setup the instance segmentation model with the given parameters.
        inst_segmentor.setup(
            epochs=args.epochs,
            loss_type=args.loss,
            gn=not args.no_gn,
            aug=not args.no_aug,
            lrd=args.lrd,
            use_amsrcr=not args.no_amsrcr,
            mbbnms=args.mbbnms,
            run_name=args.run_name,
            split_data=True
        )

        # Perform evaluation or training as specified by the user.
        if args.eval_only:

            # Get the boolean flags for the test set (use validation set if False)
            # and TTA and the inference folder name
            test_set, tta, inference_folder = args.eval_args

            # Get the model paths
            model_paths = glob(args.eval_only)

            # Loop over the pretrained model folders provided by the user
            # when evaluating.
            for mp in model_paths:
                inst_segmentor.evaluate(
                    os.path.join(mp, "model_final.pth"),
                    test_set=strtobool(test_set),
                    tta=strtobool(tta),
                    inference_folder=inference_folder,
                    use_amsrcr=not args.no_amsrcr
                )
            break
        # Perform prediction with the given pretrained model.
        elif args.predict:

            # Get the image and model paths
            img_paths, model_path = args.predict

            # Show each instance prediction overlay
            for pred in inst_segmentor.get_prediction(
                    img_paths,
                    model_path,
                    test_set=any("test" in p for p in [img_paths, model_path]),
                    use_amsrcr=not args.no_amsrcr
            ):
                plt.xticks([])
                plt.yticks([])
                plt.imshow(pred)
                plt.show()
            break
        else:
            # Train the instance segmentation model.
            inst_segmentor.train()
