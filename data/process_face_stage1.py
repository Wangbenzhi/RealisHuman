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
        [[os.path.join(dirname, 'foreground', i), os.path.join(dirname, '3ddfa', i)],
        [os.path.join(dirname, 'foreground', i), os.path.join(dirname, '3ddfa', i)]]
        for i in os.listdir(os.path.join(root_dir, dirname, '3ddfa'))
    ]
    return data_info

def process_foreground(image_path, dfa_path, save_path):
    for imgname in tqdm(os.listdir(image_path)):
        img = os.path.join(image_path, imgname)
        dfa = os.path.join(dfa_path, imgname)
        
        img = cv2.imread(img)
        dfa = cv2.imread(dfa)

        mask = np.where(np.all(dfa < 50, axis=-1), 0, 1)
        mask_3channels = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
        img = img * mask_3channels
        cv2.imwrite(os.path.join(save_path, imgname), img)
 
def process_image_pair(output_dir, input_dir, expand_gap=100):
    image_root = os.path.join(input_dir, 'image')
    dfa_root = os.path.join(input_dir, '3ddfa')
    pkl_root = os.path.join(input_dir, 'dwpose')

    for image_name in tqdm(os.listdir(image_root)):
        image_path = os.path.join(image_root, image_name)
        dfa_path = os.path.join(dfa_root, image_name)
        pkl_pose_path = get_pkl_path(image_name, pkl_root)
        if not os.path.exists(image_path) or not os.path.exists(dfa_path) or not os.path.exists(pkl_pose_path):
            continue
        pkl_pose = load_pose_data(pkl_pose_path)
        image_frame = Image.open(image_path)
        dfa_frame = Image.open(dfa_path)
        W, H = image_frame.size

        face_bbox = process_face(0, pkl_pose, W, H, expand_gap)
   
        if face_bbox:
            save_cropped_images(image_frame, dfa_frame, face_bbox, output_dir, image_name)

def get_pkl_path(image_name, pkl_root):
    prefix = image_name.split('.')[-1]
    return os.path.join(pkl_root, image_name.replace(prefix, 'pkl'))

def load_pose_data(pkl_pose_path):
    with open(pkl_pose_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)

def process_face(face_id, pkl_pose, W, H, expand_gap):

    H, W = pkl_pose['H'], pkl_pose['W']
    face = pkl_pose['faces'][0]
    
    if (face != -1).sum() <= 21:
        return None
    
    face_min, face_max = get_face_bounding_box(face, W, H)
    crop_params = get_crop_parameters(face_min, face_max, W, H, expand_gap)

    if crop_params['area'] < 400:
        return None

    return crop_params

def get_face_bounding_box(face, W, H):
    face_min = face.min(axis=0)
    face_max = face.max(axis=0)
    return face_min * [W, H], face_max * [W, H]

def get_crop_parameters(face_min, face_max, W, H, expand_gap):
    w = int(face_max[0] - face_min[0])
    h = int(face_max[1] - face_min[1])
    c_x = int((face_max[0] + face_min[0]) / 2)
    c_y = int((face_max[1] + face_min[1]) / 2)
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

def save_cropped_images(image_frame, dfa_frame, crop_params, output_dir, image_name):
    frame_path = os.path.join(output_dir, "images", f"{image_name}")
    pose_path = os.path.join(output_dir, "3ddfa", f"{image_name}")

    image_crop_frame = crop_image(image_frame, crop_params)
    dfa_crop_frame = crop_image(dfa_frame, crop_params)

    imageio.imwrite(frame_path, image_crop_frame)
    imageio.imwrite(pose_path, dfa_crop_frame)
    
if __name__ == '__main__':
    # locate and crop
    output_dir = 'data/face_example/face_chip'
    gt_dir = 'data/images'
    input_dir = 'data/images'
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '3ddfa'), exist_ok=True)
    process_image_pair(output_dir, input_dir)
    
    # process foreground
    dfa_path = 'data/face_example/face_chip/3ddfa'
    image_path = 'data/face_example/face_chip/images'
    save_path = 'data/face_example/face_chip/foreground'
    os.makedirs(save_path, exist_ok=True)
    process_foreground(image_path, dfa_path, save_path)
    
    # # get json for sampling
    root_dir = '/mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/'
    dirname = 'data/face_example/face_chip'
    output = build_taobao_dance_wo_group_stage1_val(root_dir, dirname)
    with open('data/face_example/face_stage1_val.json', 'w') as f:  
        json.dump(output, f)
