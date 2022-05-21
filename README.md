# Coordinate-Aware-Mask-R-CNN-with-Group-Normalization-Towards-Improved-Underwater-Instance-Segmentation

A hybrid Coordinate-Aware Mask R-CNN model with Group Normalization.

## Installation

1. Install the latest [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (Anaconda);
2. Download the source code and extract to the desired location;
3. Enter an Anaconda Prompt terminal window in administrator mode;
4. Navigate to the source code directory;
5. Create the **aqua** (default) conda environment by running:
  ```
  conda env create -f environment.yml
  ```
  - Alternatively, to change the default conda environment name use:
  ```
  conda env create -n <env_name> -f environment.yml
  ```
  where <env_name> should be replaced with the new custom name of the environment.

6. Run a script file (dataset and pretrained COCO weights are needed).

