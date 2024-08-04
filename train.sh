#!/bin/bash

# apt-get update -y
# apt-get install -y ffmpeg libsm6 libxext6 
# apt-get install -y  libglib2.0-0
# apt-get install -y libsm6 libxext6 libxrender1 libfontconfig1


#face task
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
    train.py --config configs/stage1_face_wbz.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     train_paste_inpainting.py --config configs/stage3_face_paste.yaml 
