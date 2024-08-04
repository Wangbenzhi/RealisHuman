from annotator.dwpose import DWposeDetector
import math
import argparse
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='', help = 'input path')
parser.add_argument('--output_path', type=str, default='', help = 'output path')
args = parser.parse_args()


if __name__ == "__main__":
    input_path = args.input_path
    assert os.path.exists(input_path)
    output_path_cur = args.output_path
    if not os.path.exists(output_path_cur):
        os.mkdir(output_path_cur)
        
    pose = DWposeDetector()
    for i, image_name in tqdm(enumerate(os.listdir(input_path))):
        test_image = os.path.join(input_path, image_name)
        oriImg = cv2.imread(test_image)  # B,G,R order
        out = pose(oriImg)
        suffixes = image_name.split('.')[-1]
        pkl_file = os.path.join(output_path_cur, image_name.replace(suffixes, 'pkl'))
        with open(pkl_file, 'wb') as file:
            pickle.dump(out, file)
