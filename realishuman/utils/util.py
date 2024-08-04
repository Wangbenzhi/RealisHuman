import os
import cv2
import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from omegaconf import OmegaConf

from omegaconf.listconfig import ListConfig
from torch.utils.data.distributed import DistributedSampler

from realishuman.data.part_chip_dataset import PartChipDataset
from realishuman.data.part_chip_dataset_paste_inpaint import PartChipDatasetPasteInpaint

DATASET_REG_DICT = {

    "PartChipDataset": PartChipDataset,
    "PartChipDatasetPasteInpaint":PartChipDatasetPasteInpaint,
}


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c f h w -> f b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = ((x + 1.0) / 2.0).clamp(0, 1)
        x = (x * 255).cpu().numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps,)


def get_distributed_dataloader(dataset_config, batch_size,
                               num_processes, num_workers, shuffle,
                               global_rank, seed, drop_last=True):
    # Get the dataset
    if isinstance(dataset_config, ListConfig):
        dataset_list = []
        for dc in dataset_config:
            dc = dc.get("dataset")
            dataset_class = DATASET_REG_DICT[dc.get("dataset_class")]
            dataset_list.append(
                dataset_class(**OmegaConf.to_container(dc.get("args")))
            )
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        dataset_class = DATASET_REG_DICT[dataset_config.get("dataset_class")]
        dataset = dataset_class(**OmegaConf.to_container(dataset_config.get("args")))
    # Get dist sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=shuffle,
        seed=seed,
    )
    # DataLoaders creation
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last, #wbz revise
    )
    return dataloader


def get_dataloader(dataset_config, batch_size, num_workers, shuffle):
    # Get the dataset
    if isinstance(dataset_config, ListConfig):
        dataset_list = []
        for dc in dataset_config:
            dc = dc.get("dataset")
            dataset_class = DATASET_REG_DICT[dc.get("dataset_class")]
            dataset_list.append(
                dataset_class(**OmegaConf.to_container(dc.get("args")))
            )
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        dataset_class = DATASET_REG_DICT[dataset_config.get("dataset_class")]
        dataset = dataset_class(**OmegaConf.to_container(dataset_config.get("args")))

    # DataLoaders creation
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def sanity_check(batch, output_dir, image_finetune, global_rank):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            for idx, (data_id, data_value) in enumerate(zip(batch["data_key"], v)):
                data_value = data_value[None, ...]
                if k == 'data_key':
                    continue
                if "pose" not in k:
                    data_value = (data_value / 2. + 0.5).clamp(0, 1)
                if image_finetune or "ref" in k:
                    torchvision.utils.save_image(
                        data_value, os.path.join(output_dir, f"{data_id}_{k}_{global_rank}.jpg"))
                else:
                    save_videos_grid(
                        data_value, os.path.join(output_dir, f"{data_id}_{k}_{global_rank}.gif"), rescale=False)
