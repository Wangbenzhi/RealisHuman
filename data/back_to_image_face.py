import os
from tqdm import tqdm
from PIL import Image
import pickle
import numpy as np
import cv2
import imageio
import argparse
from concurrent.futures import ThreadPoolExecutor

def create_mask(h, w, fade_start=150, gap_size=50, blur_size=21, center_x=None, center_y=None):
    mask = np.zeros((h, w), dtype=np.float32)
    
    if center_x is None and center_y is None:
        center_x, center_y = w // 2 , h // 2
    
    y, x = np.ogrid[:h, :w]
    
    dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    max_dist = np.sqrt((center_x-gap_size)**2 + (center_y-gap_size)**2)
    mask_region = (dist_to_center >= fade_start) & (dist_to_center < max_dist + fade_start)
    mask[mask_region] = (dist_to_center[mask_region] - fade_start) / (max_dist - fade_start)
    mask[mask_region] = 1 - np.clip(mask[mask_region], 0, 1)
    
    mask[dist_to_center < fade_start] = 1.0
    
    mask[(x < gap_size) | (x >= w - gap_size) | (y < gap_size) | (y >= h - gap_size)] = 0.0
    
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return mask

def create_fade_mask_with_layers_optimized(h, w, crop_params, target_ratio, gap_size=50, blur_size=21):
    if 'box1_cx' not in crop_params:
        w, h = crop_params['width'], crop_params['height']
        fade_start = (max(h, w)//2) * target_ratio + 70
        mask = create_mask(512, 512, fade_start)
    else:
        w, h = crop_params['box1_width'], crop_params['box1_height']
        fade_start = (max(h, w)//2) * target_ratio + 70
        mask1 = create_mask(512, 512, fade_start, center_x=crop_params['box1_cx'] * target_ratio, center_y=crop_params['box1_cy'] * target_ratio)
        
        w, h = crop_params['box2_width'], crop_params['box2_height']
        fade_start = (max(h, w)//2) * target_ratio + 70
        mask2 = create_mask(512, 512, fade_start, center_x=crop_params['box2_cx'] * target_ratio, center_y=crop_params['box2_cy'] * target_ratio)
        
        mask = np.maximum(mask1, mask2)
    return mask

def horizontal_concatenate_images(image_name):
    output_path = 'data/images/compare_results/'
    os.makedirs(output_path, exist_ok=True)
    image_path = 'data/images/image/'
    
    image1_path = image_path + image_name
    image2_path = 'data/images/results_face/' + image_name.replace('.jpg','.png')
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        return
    
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    height, width = image1.shape[:2]

    image2 = cv2.resize(image2, (width, height))
    min_height = min(image1.shape[0], image2.shape[0])
    image1 = image1[:min_height]
    image2 = image2[:min_height]
    
    concatenated_image = np.concatenate((image1, image2), axis=1)
    cv2.imwrite(output_path+image_name, concatenated_image)
    print(f"Concatenated image saved to: {output_path}")

def process_image_pair(image_name, output_dir, input_dir, sub_dir='image', expand_gap=100):
    image_root = os.path.join(output_dir, sub_dir)
    pkl_root = os.path.join(output_dir, 'dwpose')
    
    image_path = os.path.join(image_root, image_name)
    pkl_pose_path = get_pkl_path(image_name, pkl_root)

    pkl_pose = load_pose_data(pkl_pose_path)
    image_frame = Image.open(image_path)
    W, H = image_frame.size

    face_bbox = process_face(0, pkl_pose, W, H, expand_gap)

    if face_bbox:
        image_frame = save_refined_images(image_frame, face_bbox, input_dir, image_name)
    if image_frame is not None:
        results_path = os.path.join(output_dir, "results_face")
        os.makedirs(results_path, exist_ok=True)
        imageio.imwrite(os.path.join(results_path, f"{image_name}"), image_frame)

def get_pkl_path(image_name, pkl_root):
    prefix = image_name.split('.')[-1]
    return os.path.join(pkl_root, image_name.replace(prefix, 'pkl'))

def load_pose_data(pkl_pose_path):
    with open(pkl_pose_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)

def process_face(face_id, pkl_pose, W, H, expand_gap):
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
        'cx': c_x,
        'cy': c_y,
        'width': w,
        'height': h,
        'area': w*h
    }

def paste_back_image(image_frame, crop_params, result_frame):
    extend_side = crop_params['extend_side']
    t, l =  crop_params['top'], crop_params['left']
    extended_image = np.pad(
        np.array(image_frame), 
        ((extend_side*2, extend_side*2), 
         (extend_side*2, extend_side*2), 
         (0, 0))
    )
    original_height, original_width = result_frame.size[:2]
    target_ratio = original_height / (extend_side*2) 
    old_width = extended_image.shape[1]
    old_height = extended_image.shape[0]
    new_width = int(old_width * target_ratio)
    new_height = int(old_height * target_ratio)
    pad_frame_resized = Image.fromarray(extended_image).resize((new_width, new_height))
    pad_frame_resized = np.array(pad_frame_resized)
    t_resized, l_resized = int((t+extend_side*2) * target_ratio), int((l+extend_side*2) * target_ratio)
    d_resized, r_resized = t_resized + result_frame.size[0], l_resized + result_frame.size[1]
    
    blend_mask = create_fade_mask_with_layers_optimized(original_height, original_width, crop_params, target_ratio, gap_size=50)
    blend_mask_3d = np.repeat(blend_mask[:, :, np.newaxis], 3, axis=2)

    target_patch = pad_frame_resized[t_resized:d_resized, l_resized:r_resized, :]
    source_patch = np.array(result_frame)
    pad_frame_resized[t_resized:d_resized, l_resized:r_resized, :] = blend_mask_3d * source_patch + (1 - blend_mask_3d) * target_patch
    pad_frame_resized = Image.fromarray(pad_frame_resized).resize((old_width, old_height))
    pad_frame_resized = np.array(pad_frame_resized)         
    frame = pad_frame_resized[extend_side*2:-extend_side*2, extend_side*2:-extend_side*2]
    
    return frame

def save_refined_images(image_frame, crop_params, input_dir, image_name):
    inpaint_path = os.path.join(input_dir, "inpaint", f"{image_name}")
    if os.path.exists(inpaint_path):
        inpaint_frame = Image.open(inpaint_path)
        image_frame = paste_back_image(image_frame, crop_params, inpaint_frame)
        return image_frame 
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some data with a specified sub-directory.")
    parser.add_argument('--sub_dir', type=str, required=True, default='image')
    args = parser.parse_args()
    
    output_dir = 'data/images/'
    input_dir = 'data/face_example/face_chip'
    
    # Process image pairs with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        image_root = os.path.join(output_dir, args.sub_dir)
        image_names = os.listdir(image_root)
        futures = [executor.submit(process_image_pair, image_name, output_dir, input_dir, args.sub_dir) for image_name in image_names]
        for future in tqdm(futures):
            future.result()
    
    # Concatenate images with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        image_path = 'data/images/image/'
        image_names = os.listdir(image_path)
        futures = [executor.submit(horizontal_concatenate_images, image_name) for image_name in image_names]
        for future in tqdm(futures):
            future.result()
