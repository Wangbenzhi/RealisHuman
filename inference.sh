ln -s /mnt/workspace/workgroup /groupnas
source activate 


#AA tt stage1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference.py --config configs/stage1_tiktok.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_moreAA_tt/local_hand/post-processed/repair \
#             --ckpt /mnt_kg/wangbenzhi.wbz/animate_anyone/outputs/stage1_tiktok-2024-05-11T18/checkpoints/checkpoint-iter-20000.ckpt

#AA tt stage2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_paste_handinpaint.py --config configs/stage3_tiktok_paste.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_moreAA_tt/local_hand/origin/inpainted \
#             --ckpt /mnt_kg/wangbenzhi.wbz/animate_anyone/outputs/stage3_tiktok_paste-2024-05-12T00/checkpoints/checkpoint-iter-20000.ckpt

# disco tt stage1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference.py --config configs/stage1_tiktok.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_disco_tt/local_hand/origin/repair \
#             --ckpt /mnt_kg/wangbenzhi.wbz/animate_anyone/outputs/stage1_tiktok-2024-05-11T18/checkpoints/checkpoint-iter-50000.ckpt > output_0.log 2>&1 &
#disco tt stage2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_paste_handinpaint.py --config configs/stage3_tiktok_paste.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_disco_tt/local_hand/origin/inpainted \
#             --ckpt /mnt_kg/wangbenzhi.wbz/animate_anyone/outputs/stage3_tiktok_paste-2024-05-12T00/checkpoints/checkpoint-iter-20000.ckpt > output_0.log 2>&1 &



#AA ubc stage1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference.py --config configs/stage1_ubc.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_moreAA_ubc/local_hand/origin/repair \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage1_hamer_wbz-2024-04-18T21/checkpoints/checkpoint-iter-50000.ckpt

#AA ubc stage2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_paste_handinpaint.py --config configs/stage3_ubc_paste.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_moreAA_ubc/local_hand/origin/inpainted \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage3_inpainting_wbz_paste-2024-04-21T04/checkpoints/checkpoint-iter-20000.ckpt
#AA ted stage1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference.py --config configs/stage1_ted.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_moreAA_ted/local_hand/origin/repair \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage1_hamer_wbz-2024-04-18T21/checkpoints/checkpoint-iter-50000.ckpt
#AA ted stage2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_paste_handinpaint.py --config configs/stage3_ted_paste.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_moreAA_ted/local_hand/origin/inpainted \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage3_inpainting_wbz_paste-2024-04-21T04/checkpoints/checkpoint-iter-20000.ckpt


#champ ubc stage1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference.py --config configs/stage1_ubc.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_champ_ubc/local_hand/origin/repair \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage1_hamer_wbz-2024-04-18T21/checkpoints/checkpoint-iter-50000.ckpt

#champ ubc stage2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_paste_handinpaint.py --config configs/stage3_ubc_paste.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_champ_ubc/local_hand/origin/inpainted \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage3_inpainting_wbz_paste-2024-04-21T04/checkpoints/checkpoint-iter-20000.ckpt
#champ ted stage1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference.py --config configs/stage1_ted.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_champ_ted/local_hand/origin/repair \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage1_hamer_wbz-2024-04-18T21/checkpoints/checkpoint-iter-50000.ckpt
#champ ted stage2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_paste_handinpaint.py --config configs/stage3_ted_paste.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_champ_ted/local_hand/origin/inpainted \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage3_inpainting_wbz_paste-2024-04-21T04/checkpoints/checkpoint-iter-20000.ckpt


#magicanimate ubc stage1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference.py --config configs/stage1_ubc.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_magicanimate_ubc/local_hand/origin/repair \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage1_hamer_wbz-2024-04-18T21/checkpoints/checkpoint-iter-50000.ckpt > output_0.log 2>&1 &

#magicanimate ubc stage2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_paste_handinpaint.py --config configs/stage3_ubc_paste.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_magicanimate_ubc/local_hand/origin/inpainted_new \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage3_inpainting_wbz_paste-2024-04-21T04/checkpoints/checkpoint-iter-20000.ckpt > output_0.log 2>&1 &

#face repair stage1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference.py --config configs/stage1_face_wbz.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_moreAA_ubc/local_face/origin/repair \
#             --ckpt  /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage1_face_wbz-2024-07-07T08/checkpoints/checkpoint-iter-50000.ckpt  > output_0.log 2>&1 &
#face repair stage2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_paste_handinpaint.py --config configs/stage3_face_paste.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/data/Academy/methods-exp/val_moreAA_ubc/local_face/origin/inpainted \
#             --ckpt  /mnt/workspace/workgroup/wangbenzhi.wbz/animate_anyone/outputs/stage3_face_paste-2024-07-07T00/checkpoints/checkpoint-iter-40000.ckpt  > output_0.log 2>&1 &



# hand repair
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_stage1.py --config configs/stage1.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/data/hand_example/hand_chip/repair \
#             --ckpt /mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/checkpoint/stage1_hand/checkpoint-stage1-hand.ckpt  > output_0.log 2>&1 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --nnodes=1 --nproc_per_node=8 \
#     inference_stage2.py --config configs/stage2.yaml --output  /mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/data/hand_example/hand_chip/inpaint \
#             --ckpt  /mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/checkpoint/stage2_hand/checkpoint-stage2-hand.ckpt  > output_1.log 2>&1 
