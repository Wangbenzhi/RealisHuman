import json
import os
import traceback

import random

import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader


class PartChipDataset(Dataset):
    def __init__(self, root, data_info_json,
                 sample_size=(512, 512), clip_size=(224, 224),
                 group=False, ref_mode=random, split='train'):
        super().__init__()
        self.split = split
        self.root = root
        with open(data_info_json) as json_file:
            self.data_info = json.load(json_file)
        self.sample_size = sample_size
        self.clip_size = clip_size
        self.group = group
        self.ref_mode = ref_mode

        self.length = len(self.data_info)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                size=self.sample_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                size=self.clip_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.Normalize([0.485, 0.456, 0.406],     # used for dino
                                 [0.229, 0.224, 0.225]),    # used for dino
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

    def get_batch(self, idx):
        meta_info = self.data_info[idx]
        if self.group:
            if self.split == 'train':
                ref_info, img_info = random.sample(meta_info, 2)
            elif self.split == 'val':
                ref_info, img_info = meta_info[0], meta_info[1]
            ref_img_path, ref_pose_path = os.path.join(self.root, ref_info[0]), os.path.join(self.root, ref_info[1])
        else:
            img_info = meta_info
            ref_img_path = ref_pose_path = None
        img_path, pose_path = os.path.join(self.root, img_info[0]), os.path.join(self.root, img_info[1])

        image = Image.open(img_path)
        pose = Image.open(pose_path)
        _ref_image = None if ref_img_path is None else Image.open(ref_img_path)
        ref_pose = None if ref_pose_path is None else Image.open(ref_pose_path)

        image = self.img_transform(image)
        pose = self.pose_transform(pose)
        if _ref_image is not None and ref_pose is not None:
            ref_image = self.img_transform(_ref_image)
            ref_image_clip = self.clip_transform(_ref_image)
            ref_pose = self.pose_transform(ref_pose)
        else:
            ref_image = ref_image_clip = ref_pose = None

        data_key = img_info[0].split('/')[-1]
        return {"data_key": data_key, "image": image, "pose": pose, "ref_image": ref_image,
                "ref_pose": ref_pose, "ref_image_clip": ref_image_clip}

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
    save_dir = "./debug_image_dataset/img/"
    os.makedirs(save_dir, exist_ok=True)
    val_data = PartChipDataset(
        root='/mnt/workspace/workgroup/wangbenzhi.wbz/data/',
        data_info_json='/mnt/workspace/workgroup/wangbenzhi.wbz/data/face_chip/face_stage1_train.json',
        group=True, split='val')
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False, num_workers=1)
    for i, sample in enumerate(val_loader):
        print(sample["data_key"])
        print(f"image shape is {sample['image'].shape}")
        print(f"pose shape is {sample['pose'].shape}")
        save_obj = torch.cat([
            (sample["image"].cpu() / 2 + 0.5).clamp(0, 1),
            (sample["ref_image"].cpu() / 2 + 0.5).clamp(0, 1),
            sample["pose"].cpu()]
        , dim=-1)
        save_image(save_obj, save_dir + f"sample_{i}.png")
        if i > 2:
            break
