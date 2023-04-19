#!/bin/bash
conda create --name deq-flow-former python==3.9
conda activate deq-flow-former
conda install pytorch torchvision torchaudio -c pytorch
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install matplotlib tensorboard scipy opencv-python einops termcolor
pip install yacs loguru einops timm imageio
