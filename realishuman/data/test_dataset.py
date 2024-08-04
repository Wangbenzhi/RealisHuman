import os
import os.path as osp
import torch
import random

from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, split, data_lst_file_path_lst,
                 sample_size=(960, 640), clip_size=(420, 280),
                 ref_mode='random'):
        super().__init__()

        self.split = split
        self.sample_size = sample_size
        self.clip_size = clip_size
        self.data_lst = []
        for data_lst_file_path in data_lst_file_path_lst:
            self.data_lst += open(data_lst_file_path).readlines()
        self.length = len(self.data_lst)
        self.ref_mode = ref_mode

        print(f"{self.split} dataset length is {self.length}")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                size=self.sample_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=None,
            ),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                size=self.clip_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=None,
            ),
            transforms.Normalize([0.485, 0.456, 0.406],     # used for dino
                                 [0.229, 0.224, 0.225]),    # used for dino
        ])
        self.pose_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                size=self.sample_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=None,
            ),
        ])

    def get_file_path(self, cur_img_dir, ref_img_key, tgt_img_key):
        img_folder_key = '/img/'
        mask_folder_key = '/mask/'
        ref_img_path = osp.join(cur_img_dir, ref_img_key)
        tgt_img_path = osp.join(cur_img_dir, tgt_img_key)
        ref_pose_path = '/groupnas/zhoujingkai.zjk/data/AIGC' + ref_img_path.split('AIGC')[1]
        ref_mask_path = ref_img_path.replace(img_folder_key, mask_folder_key).replace('.jpg', '.png')
        tgt_pose_path = '/groupnas/zhoujingkai.zjk/data/AIGC' + tgt_img_path.split('AIGC')[1]

        return ref_img_path, tgt_img_path, ref_pose_path, tgt_pose_path, ref_mask_path

    def __len__(self):
        return len(self.data_lst)

    def get_metadata(self, idx):
        idx = idx % self.length

        data_info = self.data_lst[idx].rstrip().split(',')
        cur_img_dir = data_info[0]
        cur_vid_key = cur_img_dir.split('/')[-1]
        img_key_lst = data_info[1:]
        ref_img_key = img_key_lst[0]
        tgt_img_key = random.SystemRandom().sample(img_key_lst[1:], 1)[0] \
            if self.split == 'train' else img_key_lst[1:][len(img_key_lst[1:])//2]
        ref_img_path, tgt_img_path, ref_pose_path, tgt_pose_path, ref_mask_path = self.get_file_path(
            cur_img_dir, ref_img_key, tgt_img_key)

        ref_img = Image.open(ref_img_path).convert('RGB')
        tgt_img = Image.open(tgt_img_path).convert('RGB')
        ref_pose_img = Image.open(ref_pose_path).convert('RGB')
        tgt_pose_img = Image.open(tgt_pose_path).convert('RGB')

        # preparing outputs
        meta_data = {}
        meta_data['dataset_name'] = ref_img_path.split('/')[5]
        meta_data['img_key'] = f"{cur_vid_key}_{ref_img_key}_{tgt_img_key}"
        meta_data['ref_img'] = ref_img
        meta_data['ref_dino_img'] = ref_img
        meta_data['tgt_img'] = tgt_img
        meta_data['ref_pose_img'] = ref_pose_img
        meta_data['tgt_pose_img'] = tgt_pose_img

        return meta_data

    @staticmethod
    def augmentation(frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame)

    def __getitem__(self, idx):
        try:
            raw_data = self.get_metadata(idx)
        except:
            print(self.data_lst[idx])
            print('打开 idx 图片失败，重新尝试打开 0th图片')
            raw_data = self.get_metadata(0)

        img_key = raw_data['img_key']
        ref_img = raw_data['ref_img']
        ref_dino_img = raw_data['ref_dino_img']
        tgt_img = raw_data['tgt_img']
        ref_pose_img = raw_data['ref_pose_img']
        tgt_pose_img = raw_data['tgt_pose_img']

        # DA
        state = torch.get_rng_state()
        ref_img = self.augmentation(ref_img, self.img_transform, state)
        tgt_img = self.augmentation(tgt_img, self.img_transform, state)
        ref_pose_img = self.augmentation(ref_pose_img, self.pose_transform, state)
        ref_dino_img = self.augmentation(ref_dino_img, self.clip_transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_img, self.pose_transform, state)

        return {"data_key": img_key, "image": tgt_img, "pose": tgt_pose_img, "ref_image": ref_img,
                "ref_pose": ref_pose_img, "ref_image_clip": ref_dino_img}


if __name__ == "__main__":
    save_dir = "./debug_image_dataset/img/"
    os.makedirs(save_dir, exist_ok=True)
    val_data = ImageDataset(
        split="train",
        data_lst_file_path_lst=[
          '/mnt_dy/yingtian/project/AIGC/taobao_pose_transfer/data_lst/data_lst_dwpose_body_hand_0105/'
          'train_姿态编辑图片数据20131225.txt',
          '/mnt_dy/yingtian/project/AIGC/taobao_pose_transfer/data_lst/data_lst_dwpose_body_hand_0105/'
          'train_姿态编辑图片.txt',
          '/mnt_dy/yingtian/project/AIGC/taobao_pose_transfer/data_lst/data_lst_dwpose_body_hand_0105/'
          'train_淘宝主图视频.txt',
          '/mnt_dy/yingtian/project/AIGC/taobao_pose_transfer/data_lst/data_lst_dwpose_body_hand_0105/'
          'train_dress1.txt',
          '/mnt_dy/yingtian/project/AIGC/taobao_pose_transfer/data_lst/data_lst_dwpose_body_hand_0105/'
          'train_dress2.txt',
          '/mnt_dy/yingtian/project/AIGC/taobao_pose_transfer/data_lst/data_lst_dwpose_body_hand_0105/'
          'train_deepfashion.txt',
          '/mnt_dy/yingtian/project/AIGC/taobao_pose_transfer/data_lst/data_lst_dwpose_body_hand_0105/'
          'train_UBC.txt',
        ])
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False, num_workers=1)
    for i, sample in enumerate(val_loader):
        print(sample["data_key"])
        print(f"image shape is {sample['image'].shape}")
        print(f"pose shape is {sample['pose'].shape}")
        print(f"ref_image shape is {sample['ref_image'].shape}")
        print(f"ref_pose shape is {sample['ref_pose'].shape}")
        save_image(sample["image"]/2+0.5, save_dir + f"image_{i}.png")
        save_image(sample["pose"], save_dir + f"pose_{i}.png")
        save_image(sample["ref_image"]/2+0.5, save_dir + f"ref_image_{i}.png")
        save_image(sample["ref_image_clip"]/2+0.5, save_dir + f"ref_image_clip{i}.png")
        save_image(sample["ref_pose"], save_dir + f"ref_pose_{i}.png")
        if i > 10:
            break
