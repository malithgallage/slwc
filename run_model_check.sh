#!/bin/sh
module load tensorflow/nvidia-19.11-tf2-py3
pip install --user scikit-image segmentation_models
# srun --account=Project_2005434 --partition=gpu --gres=gpu:v100:1 --mem=5G python model_check.py -n 0
srun --account=Project_2005434 --partition=gpu --gres=gpu:v100:1 --mem=5G python model_check.py -n 45 
# srun --account=Project_2005434 --partition=gpu --gres=gpu:v100:1 --mem=5G python model_check.py -n 551 
# srun --account=Project_2005434 --partition=gpu --gres=gpu:v100:1 --mem=5G python model_check.py -n 5
