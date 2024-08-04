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


def build_taobao_dance_wo_group(root_dir, dirname):
    data_info = []
    for i in os.listdir(os.path.join(root_dir, dirname, 'repair')):
        image_path = os.path.join(dirname, 'images', i)
        hamer_path = os.path.join(dirname , 'hamers', i)
        repair_path = os.path.join(dirname, 'repair', i)
        if os.path.exists(os.path.join(root_dir,image_path)) and os.path.exists(os.path.join(root_dir,hamer_path)) and os.path.exists(os.path.join(root_dir,repair_path)):
            data_info.append([image_path, hamer_path, repair_path])
    print(len(data_info))
    return data_info


if __name__ == '__main__':
    
    root_dir = '/mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/'
    dirname = 'data/hand_example/hand_chip'
    output = build_taobao_dance_wo_group(root_dir, dirname)
    with open('data/hand_example/hand_stage2_val.json', 'w') as f:  
        json.dump(output, f)

    