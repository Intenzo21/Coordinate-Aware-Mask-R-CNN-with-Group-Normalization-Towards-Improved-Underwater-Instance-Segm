"""
Script plotting the data distribution across the training, validation and test sets.
"""

from utils import plot_dist_argument_parser, plot_distributions
from glob import glob

if __name__ == "__main__":

    # Parse the command line arguments
    args = plot_dist_argument_parser().parse_args()

    # Plot the training, validation and test set instances per
    # category and image distributions.
    plot_distributions(
        train_val_paths=glob(args.train_val_paths),
        test_paths=glob(args.test_paths)
    )
