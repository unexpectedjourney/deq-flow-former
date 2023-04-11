#!/bin/bash
conda create --name deq-flow-former
conda activate deq-flow-former
conda install pytorch torchvision torchaudio -c pytorch
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib tensorboard scipy opencv einops termcolor -c conda-forge
pip install yacs loguru einops timm imageio
