import os 
from tqdm import tqdm
from PIL import Image
import pickle
import numpy as np
import cv2
import json
import re
import shutil
import json
import math
import imageio
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_taobao_dance_wo_group_stage1_val(root_dir, dirname):
    data_info = [
        [[os.path.join(dirname, 'foreground', i), os.path.join(dirname, 'hamers', i)],
        [os.path.join(dirname, 'foreground', i), os.path.join(dirname, 'hamers', i)]]
        for i in os.listdir(os.path.join(root_dir, dirname, 'hamers'))
    ]
    return data_info

def process_foreground(image_path, hamer_path, save_path):
    for imgname in tqdm(os.listdir(image_path)):
        img = os.path.join(image_path, imgname)
        hamer = os.path.join(hamer_path, imgname)
        
        img = cv2.imread(img)
        hamer = cv2.imread(hamer)

        mask = np.where(np.all(hamer < 100, axis=-1), 0, 1)
        mask_3channels = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
        img = img * mask_3channels
        cv2.imwrite(os.path.join(save_path, imgname), img)
 
def process_image_pair(output_dir, input_dir, expand_gap=100):
    image_root = os.path.join(input_dir, 'image')
    hamer_root = os.path.join(input_dir, 'hamer')
    pkl_root = os.path.join(input_dir, 'dwpose')

    for image_name in tqdm(os.listdir(image_root)):
        image_path = os.path.join(image_root, image_name)
        hamer_path = os.path.join(hamer_root, image_name)
        pkl_pose_path = get_pkl_path(image_name, pkl_root)
        if not os.path.exists(image_path) or not os.path.exists(hamer_path) or not os.path.exists(pkl_pose_path):
            continue
        pkl_pose = load_pose_data(pkl_pose_path)
        image_frame = Image.open(image_path)
        hamer_frame = Image.open(hamer_path)
        W, H = image_frame.size

        left_hand_bbox = process_hand(0, pkl_pose, W, H, expand_gap)
        right_hand_bbox = process_hand(1, pkl_pose, W, H, expand_gap)

        if left_hand_bbox and right_hand_bbox and check_intersection(left_hand_bbox, right_hand_bbox):
            merged_bbox = merge_bounding_boxes(left_hand_bbox, right_hand_bbox)
            save_cropped_images(image_frame, hamer_frame, merged_bbox, output_dir, image_name)
        else:
            if left_hand_bbox:
                save_cropped_images(image_frame, hamer_frame, left_hand_bbox, output_dir, image_name, 'l')
            if right_hand_bbox:
                save_cropped_images(image_frame, hamer_frame, right_hand_bbox, output_dir, image_name, 'r')

def get_pkl_path(image_name, pkl_root):
    prefix = image_name.split('.')[-1]
    return os.path.join(pkl_root, image_name.replace(prefix, 'pkl'))

def load_pose_data(pkl_pose_path):
    with open(pkl_pose_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)

def process_hand(hand_id, pkl_pose, W, H, expand_gap):
    hand_score = pkl_pose['hands_score'][hand_id]
    hand = pkl_pose['hands'][hand_id]
    valid_indices = hand_score >= 0.5
    # hand[~valid_indices] = -1

    if valid_indices.sum() <= 10:
        return None

    hand_min, hand_max = get_hand_bounding_box(hand, W, H)
    crop_params = get_crop_parameters(hand_min, hand_max, W, H, expand_gap)

    if crop_params['area'] < 400:
        return None

    return crop_params

def get_hand_bounding_box(hand, W, H):
    hand_min = hand.min(axis=0)
    hand_max = hand.max(axis=0)
    return hand_min * [W, H], hand_max * [W, H]

def get_crop_parameters(hand_min, hand_max, W, H, expand_gap):
    w = int(hand_max[0] - hand_min[0])
    h = int(hand_max[1] - hand_min[1])
    c_x = int((hand_max[0] + hand_min[0]) / 2)
    c_y = int((hand_max[1] + hand_min[1]) / 2)
    extend_side = max(h, w) // 2 + expand_gap

    return {
        'left': c_x - extend_side,
        'right': c_x + extend_side,
        'top': c_y - extend_side,
        'bottom': c_y + extend_side,
        'extend_side': extend_side,
        'area': w * h,
        'width': w,
        'height': h
    }

def check_intersection(bbox1, bbox2):
    x_left = max(bbox1['left'], bbox2['left'])
    y_top = max(bbox1['top'], bbox2['top'])
    x_right = min(bbox1['right'], bbox2['right'])
    y_bottom = min(bbox1['bottom'], bbox2['bottom'])

    intersection_width = x_right - x_left
    intersection_height = y_bottom - y_top

    if intersection_width > 0 and intersection_height > 0:
        return True
    else:
        return False

def merge_bounding_boxes(bbox1, bbox2):
    left = min(bbox1['left'], bbox2['left'])
    right = max(bbox1['right'], bbox2['right'])
    top = min(bbox1['top'], bbox2['top'])
    bottom = max(bbox1['bottom'], bbox2['bottom'])

    width = right - left
    height = bottom - top
    extend_side = max(width, height) // 2

    c_x = (left + right) // 2
    c_y = (top + bottom) // 2

    return {
        'left': c_x - extend_side,
        'right': c_x + extend_side,
        'top': c_y - extend_side,
        'bottom': c_y + extend_side,
        'extend_side': extend_side,
        'area': width * height
    }

def crop_image(image_frame, crop_params):
    extended_image = np.pad(
        np.array(image_frame), 
        ((crop_params['extend_side']*2, crop_params['extend_side']*2), 
         (crop_params['extend_side']*2, crop_params['extend_side']*2), 
         (0, 0))
    )
    return extended_image[
        crop_params['top'] + crop_params['extend_side']*2 : crop_params['bottom'] + crop_params['extend_side']*2,
        crop_params['left'] + crop_params['extend_side']*2 : crop_params['right'] + crop_params['extend_side']*2
    ]

def save_cropped_images(image_frame, hamer_frame, crop_params, output_dir, image_name, hand_label='merged'):
    frame_path = os.path.join(output_dir, "images", f"{image_name[:-4]}_{hand_label}.jpg")
    pose_path = os.path.join(output_dir, "hamers", f"{image_name[:-4]}_{hand_label}.jpg")

    image_crop_frame = crop_image(image_frame, crop_params)
    hamer_crop_frame = crop_image(hamer_frame, crop_params)

    imageio.imwrite(frame_path, image_crop_frame)
    imageio.imwrite(pose_path, hamer_crop_frame)
    
if __name__ == '__main__':
    # locate and crop
    output_dir = 'data/hand_example/hand_chip'
    gt_dir = 'data/images'
    input_dir = 'data/images'
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'hamers'), exist_ok=True)
    process_image_pair(output_dir, input_dir)
    # process foreground
    hamer_path = 'data/hand_example/hand_chip/hamers'
    image_path = 'data/hand_example/hand_chip/images'
    save_path = 'data/hand_example/hand_chip/foreground'
    os.makedirs(save_path, exist_ok=True)
    process_foreground(image_path, hamer_path, save_path)
    # get json for sampling
    root_dir = '/mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/'
    dirname = 'data/hand_example/hand_chip'
    output = build_taobao_dance_wo_group_stage1_val(root_dir, dirname)
    with open('data/hand_example/hand_stage1_val.json', 'w') as f:  
        json.dump(output, f)
