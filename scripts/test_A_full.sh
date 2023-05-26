#!/bin/bash

# python3 -u main.py --eval --name deq-flow-A-sintel-eval --stage sintel \
#     --validation sintel --restore_ckpt checkpoints/deq-flow-A-sintel-run-1.pth \
#     --gpus 0 1 --batch_size 12 --image_size 368 768 --wdecay 0.0001 --gamma=0.90 \
#     --wnorm --f_solver anderson \
#     --f_thres 36 --n_losses 6 --phantom_grad 3

python3 -u main.py --test --name deq-flow-A-sintel-test --stage sintel \
    --test_set sintel kitti --restore_ckpt checkpoints/deq-flow-A-sintel-run-1.pth \
    --gpus 0 1 --batch_size 12 --image_size 368 768 --wdecay 0.0001 --gamma=0.90 \
    --wnorm --f_solver anderson \
    --f_thres 36 --n_losses 6 --phantom_grad 3 --output_path "warm_sintel_submissions_3"

# python3 -u main.py --eval --name deq-flow-A-kitti-eval --stage kitti \
#     --validation sintel kitti --restore_ckpt checkpoints/deq-flow-A-kitti-run-1.pth \
#     --gpus 0 1 --batch_size 12 --image_size 288 960 --wdecay 0.0001 --gamma=0.90 \
#     --wnorm --f_solver anderson \
#     --f_thres 36 --n_losses 6 --phantom_grad 1
# 
# python3 -u main.py --test --name deq-flow-A-kitti-test --stage kitti \
#     --test_set sintel kitti --restore_ckpt checkpoints/deq-flow-A-kitty-run-1.pth \
#     --gpus 0 1 --batch_size 12 --image_size 288 960 --wdecay 0.0001 --gamma=0.90 \
#     --wnorm --f_solver anderson \
#     --f_thres 36 --n_losses 6 --phantom_grad 1
