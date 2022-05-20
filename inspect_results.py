"""
Script used for inspecting loss results along with validation and
test set output metrics of a model.
"""
from utils import inspect_results_argument_parser, plot_losses, \
    save_infer_results, save_val_results

if __name__ == "__main__":

    # Parse the command line arguments
    args = inspect_results_argument_parser().parse_args()

    # Plot the training and validation loss graph
    if args.plot_losses:
        plot_losses(*args.plot_losses)

    # Save the inference results stored in the run
    # output text files of the model
    if args.save_txt:
        save_infer_results(*args.save_txt)

    # Save the inference results stored in the run
    # output JSON files of the model (typically validation set)
    if args.save_json:
        save_val_results(*args.save_json)
