import json
import os
import traceback
import cv2
import torch.nn.functional as F

import random
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader


class PartChipDatasetPasteInpaint(Dataset):
    def __init__(self, root, data_info_json,
                 sample_size=(512, 512), mask_gap=100, split='train', mask_thr=0.4):
        super().__init__()
        self.split = split
        self.root = root
        with open(data_info_json) as json_file:
            self.data_info = json.load(json_file)
        self.sample_size = sample_size
        self.mask_gap = mask_gap
        # self.use_hp = use_hp
        self.length = len(self.data_info)
        self.mask_thr = mask_thr
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                size=self.sample_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.pose_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                size=self.sample_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ])

    def __len__(self):
        return self.length
    
    def calculate_max_bounding_rect_from_mask(self, mask_3channels):
        # 
        mask_single_channel = mask_3channels[:, :, 0]

        contours, _ = cv2.findContours(mask_single_channel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_width = 0
        max_height = 0
        max_rect = None

        # 计算每个轮廓的外接矩形，并找到最大的一个
        for contour in contours:
            rect = cv2.boundingRect(contour)
            _, _, width, height = rect
            if width * height > max_width * max_height:
                max_width = width
                max_height = height
                max_rect = rect

        return max_width if max_width > max_height else max_height
    def process_foreground(self, pose, img, generated_part, split='train'):

        mask = torch.where(torch.all(pose < self.mask_thr, dim=0, keepdim=True), torch.tensor(0.), torch.tensor(1.)).numpy()
        mask_3channels = np.repeat(mask, img.shape[0], axis=0)
        mask_3channels = mask_3channels.transpose(1, 2, 0)
        
        kernel = np.ones((self.mask_gap, self.mask_gap,), dtype=np.float32)
        mask_3channels_dilated = cv2.dilate(mask_3channels, kernel, 1)
        mask_3channels_dilated_finalstep = mask_3channels_dilated.copy() #numpy deep copy

        length = self.calculate_max_bounding_rect_from_mask(mask_3channels)
        erosion_size=int(length*0.06)
        # erosion_size=1
        kernel_eros = np.ones((erosion_size, erosion_size), dtype=np.float32)
        mask_3channels_eroded = cv2.erode(mask_3channels, kernel_eros, iterations=1)
        mask_3channels_dilated[(mask_3channels_eroded==1)] = 0
        
        mask_3channels_eroded = mask_3channels_eroded.transpose(2, 0, 1)
        mask_3channels_eroded = torch.from_numpy(mask_3channels_eroded)    

        mask_3channels_dilated_finalstep = mask_3channels_dilated_finalstep.transpose(2, 0, 1)
        mask_3channels_dilated = mask_3channels_dilated.transpose(2, 0, 1)

        mask_3channels_dilated_finalstep = torch.from_numpy(mask_3channels_dilated_finalstep)
        mask_3channels_dilated = torch.from_numpy(mask_3channels_dilated)
        
        img_background_finalstep = img * (1 - mask_3channels_dilated_finalstep)
        img_background = img * (1 - mask_3channels_dilated)
        if generated_part is not None: 
            img_background[mask_3channels_eroded==1] = generated_part[mask_3channels_eroded==1]

        return mask_3channels_dilated[0], mask_3channels_dilated_finalstep[0], img_background_finalstep, img_background , mask_3channels_eroded      
    


    def get_batch(self, idx):
        meta_info = self.data_info[idx]

        try:
            img_path, pose_path, foreground_path = os.path.join(self.root, meta_info[0]), os.path.join(self.root, meta_info[1]), os.path.join(self.root, meta_info[2])
        except:
            img_path, pose_path, foreground_path = os.path.join(self.root, meta_info[0]), os.path.join(self.root, meta_info[1]), ""
            
        image = Image.open(img_path)
        pose = Image.open(pose_path)
        
        if self.split == "val":
            generated_hand = Image.open(foreground_path)
            generated_hand = self.img_transform(generated_hand) 
        else:
            assert foreground_path == ""
            generated_hand = None
        image = self.img_transform(image)
        pose = self.pose_transform(pose)

        mask, mask_finalstep, bg_image_finalstep, bg_image, mask_eroded = self.process_foreground(pose, image, generated_hand, self.split)

        imgname = meta_info[0].split('/')[-1]
        return {"data_key": imgname, "image": image, "bg_image": bg_image, "mask": mask, "mask_finalstep":mask_finalstep, "bg_image_finalstep":bg_image_finalstep, "mask_eroded":mask_eroded}

    def __getitem__(self, idx):
        try_cnt = 0
        while True:
            try:
                try_cnt += 1
                if try_cnt > 10:
                    break
                return self.get_batch(idx)
            except Exception as e:
                print(f"read idx: {idx} error, {type(e).__name__}: {e}")
                print(traceback.format_exc())
                idx = random.randint(0, self.length - 1)


if __name__ == "__main__":
    save_dir = "./debug/AAAI_framework/"
    os.makedirs(save_dir, exist_ok=True)
    val_data = PartChipDatasetPasteInpaint(
        root= '/mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/',
        data_info_json='data/hand_example/hand_stage2_val.json', split='val',mask_thr=0.2)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)
    for i, sample in enumerate(val_loader):

        save_image((sample["bg_image"].cpu() / 2 + 0.5).clamp(0, 1), save_dir + f"sample_{i}.png")
        save_image(sample["mask"].cpu(), save_dir + f"mask_dilated{i}.png")
        save_image(sample["mask_eroded"].cpu(), save_dir + f"mask_eroded{i}.png")
        if i > 10:
            break

    