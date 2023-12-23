# Palette Detection using OpenCV

## How To Run

- Create the environment from the environment.yml file:
  ```
  conda env create -f environment.yml
  ```
The first line of the yml file sets the new environment's name.
- Activate conda environment using
  ```
    conda activate palette
  ```
  or change the `palette` to the name of your environment name inside `environment.yml` first line
- configure the `model_dir` for tensorflow classification task and `model_yolo_dir` variable for object detection task using YOLOv8 inside `main.py` file to the directory to your liking
- run
  ```
    python main.py
  ```
