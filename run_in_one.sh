#!/bin/bash
source activate ../dwpose_env/
cd submodules/DWPose
python ControlNet-v1-1-nightly/dwpose_infer_example.py --input_path /mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/data/images/image --output_path /mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/data/images/dwpose

cd -
source activate ../hamer_env/
cd submodules/hamer-main
python demo_image.py --img_folder /mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/data/images/image --out_folder /mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/data/images/hamer --full_frame

cd -
source activate ../3ddfa_env/
cd submodules/3DDFA-V3/
python demo_dir.py --inputpath /mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/data/images/image --savepath /mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/data/images/3ddfa --device cuda --iscrop 1 --detector retinaface --ldm68 0 --ldm106 0 --ldm106_2d 0 --ldm134 0 --seg_visible 0 --seg 0 --useTex 0 --extractTex 0 --backbone resnet50

cd -
source activate ../mooreAA_env/

python data/process_hand_stage1.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 \
    inference_stage1.py --config configs/stage1-hand.yaml --output data/hand_example/hand_chip/repair \
    --ckpt checkpoint/stage1_hand/checkpoint-stage1-hand.ckpt

python data/process_hand_stage2.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 \
    inference_stage2.py --config configs/stage2-hand.yaml --output data/hand_example/hand_chip/inpaint \
    --ckpt checkpoint/stage2_hand/checkpoint-stage2-hand.ckpt

python data/back_to_image_hand.py
#======================================= Face Refinement (optional) ====================================================================#
# python data/process_face_stage1.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_stage1.py --config configs/stage1-face.yaml --output data/face_example/face_chip/repair \
#     --ckpt checkpoint/stage1_face/checkpoint-stage1-face.ckpt

# python data/process_face_stage2.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_stage2.py --config configs/stage2-face.yaml --output data/face_example/face_chip/inpaint \
#     --ckpt checkpoint/stage2_face/checkpoint-stage2-face.ckpt

# python data/back_to_image_face.py --sub_dir results_hand