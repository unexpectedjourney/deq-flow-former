#!/bin/bash

python -u main.py --total_run 1 --start_run 1 --name deq-flow-H-naive-120k-S-36-1-1 \
    --stage sintel --validation sintel \
    --gpus 0 1 2 --num_steps 120000 --eval_interval 20000 \
    --batch_size 12 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
    --f_thres 36 --f_solver naive_solver \
    --n_losses 1 --phantom_grad 1 \
    --huge --wnorm
