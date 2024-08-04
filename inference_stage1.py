import os
import logging
import argparse
import subprocess

from tqdm.auto import tqdm
from omegaconf import OmegaConf
from typing import Dict

import torch
import torchvision
import torch.distributed as dist

from transformers import AutoModel

from diffusers import AutoencoderKL, DDIMScheduler, AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from realishuman.models.realishuman_unet import RealisHumanUnet
from realishuman.pipelines.pipeline_stage1 import StageOnePipeline
from realishuman.utils.util import get_distributed_dataloader, save_videos_grid, sanity_check


def init_dist(launcher="slurm", backend="nccl", port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == "pytorch":
        rank = int(os.environ["RANK"])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == "slurm":
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        port = os.environ.get("PORT", port)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend)
        print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; "
              f"node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")

    else:
        raise NotImplementedError(f"Not implemented launcher type: `{launcher}`!")

    return local_rank


def main(
    image_finetune: bool,
    launcher: str,

    output_dir: str,
    pretrained_model_path: str,
    pretrained_clip_path: str,

    validation_data: Dict,
    unet_checkpoint_path,
    validation_kwargs: Dict = None,
    save_vid: bool = False,
    fps: int = 8,

    pretrained_vae_path: str = "",
    unet_additional_kwargs: Dict = None,
    pose_guider_kwargs: Dict = None,
    fusion_blocks: str = "full",
    clip_projector_kwargs: Dict = None,
    fix_ref_t: bool = False,
    zero_snr: bool = False,
    ref_mean_ratio: float = 0.0,
    v_pred: bool = False,
    vae_slicing: bool = False,
    
    num_workers: int = 4,
    validation_batch_size: int = 1,
    gradient_checkpointing: bool = False,

    mixed_precision_inference: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,
    global_seed: int = 42,
    is_debug: bool = False,
    sanity_check_during_validation: bool = False,
    
    *args,
    **kwargs,
):

    # Initialize distributed training
    local_rank = init_dist(launcher=launcher)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0
    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}", exist_ok=True)

    # Load scheduler, tokenizer and models
    if is_main_process:
        logging.info("Load scheduler, tokenizer and models.")
    if pretrained_vae_path != "":
        vae = AutoencoderKL.from_pretrained(pretrained_vae_path, subfolder="sd-vae-ft-mse")
    else:
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    image_encoder = AutoModel.from_pretrained(pretrained_clip_path)

    if zero_snr:
        if is_main_process:
            logging.info("Enable Zero-SNR")
        if v_pred:
            noise_scheduler = DDIMScheduler.from_pretrained(
                pretrained_model_path, subfolder="scheduler",
                prediction_type="v_prediction",
                timestep_spacing="linspace",
                rescale_betas_zero_snr=True)
        else:
            noise_scheduler = DDIMScheduler.from_pretrained(
                pretrained_model_path, subfolder="scheduler",
                rescale_betas_zero_snr=True)
    else:
        noise_scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_path, subfolder="scheduler")

    unet = RealisHumanUnet(
        pretrained_model_path=pretrained_model_path,
        image_finetune=image_finetune,
        unet_additional_kwargs=unet_additional_kwargs,
        pose_guider_kwargs=pose_guider_kwargs,
        clip_projector_kwargs=clip_projector_kwargs,
        fix_ref_t=fix_ref_t,
        fusion_blocks=fusion_blocks,
    )

    # Load pretrained unet weights
    if is_main_process:
        logging.info(f"from checkpoint: {unet_checkpoint_path}")
    unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
    if "global_step" in unet_checkpoint_path:
        if is_main_process:
            logging.info(f"global_step: {unet_checkpoint_path['global_step']}")
    state_dict = unet_checkpoint_path["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[7:]
        else:
            new_k = k
        new_state_dict[new_k] = state_dict[k]
    m, u = unet.load_state_dict(new_state_dict, strict=False)
    if is_main_process:
        logging.info(f"Load from checkpoint with missing keys:\n{m}")
        logging.info(f"Load from checkpoint with unexpected keys:\n{u}")
        assert len(u) == 0

    # Freeze vae and image_encoder
    vae.eval()
    vae.requires_grad_(False)
    image_encoder.eval()
    image_encoder.requires_grad_(False)
    unet.eval()
    unet.requires_grad_(False)

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Set validation pipeline
    validation_pipeline = StageOnePipeline(
        unet=unet, vae=vae, image_encoder=image_encoder, scheduler=noise_scheduler)
    validation_pipeline.image_finetune = image_finetune
    validation_kwargs_container = {} if validation_kwargs is None else OmegaConf.to_container(validation_kwargs)

    # move to cuda
    vae.to(local_rank)
    image_encoder.to(local_rank)
    unet.to(local_rank)
    validation_pipeline = validation_pipeline.to(local_rank)

    # Get the validation dataloader
    validation_dataloader = get_distributed_dataloader(
        dataset_config=validation_data,
        batch_size=validation_batch_size,
        num_processes=num_processes,
        num_workers=num_workers,
        shuffle=False,
        global_rank=global_rank,
        seed=global_seed,
        drop_last=False)

    if is_main_process:
        logging.info("***** Running validation *****")
        logging.info(f"  Instantaneous validation batch size per device = {validation_batch_size}")

    generator = torch.Generator(device=unet.device)
    generator.manual_seed(global_seed)
    for val_batch in tqdm(validation_dataloader):
        # check sanity during validation
        if sanity_check_during_validation:
            if is_main_process:
                os.makedirs(f"{output_dir}/sanity_check/", exist_ok=True)
            sanity_check(val_batch, f"{output_dir}/sanity_check", image_finetune, global_rank)

        height, width = val_batch["pose"].shape[-2:]
        if isinstance(val_batch["image"], torch.Tensor):
            val_gt = val_batch["image"].to(local_rank)
        val_pose = val_batch["pose"].to(local_rank)
        val_ref_image = val_batch["ref_image"].to(local_rank)
        val_ref_pose = val_batch["ref_pose"].to(local_rank)
        val_ref_image_clip = val_batch["ref_image_clip"].to(local_rank)

        with torch.cuda.amp.autocast(enabled=mixed_precision_inference):
            sample = validation_pipeline(
                pose=val_pose,
                ref_image=val_ref_image,
                ref_pose=val_ref_pose,
                ref_image_clip=val_ref_image_clip,
                height=height, width=width,
                ref_mean_ratio=ref_mean_ratio,
                **validation_kwargs_container).videos
        # TODO: support more images per prompt
        num_images_per_prompt = 1
        for idx, data_id in enumerate(val_batch["data_key"]):
            samples = sample[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
            video_length = samples.shape[2]
            val_poses = val_pose[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
            ref_images = val_ref_image[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
            if not image_finetune:
                ref_images = ref_images.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
            if isinstance(val_batch["image"], torch.Tensor):
                val_gts = val_gt[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
                save_obj = torch.cat([
                    (ref_images.cpu() / 2 + 0.5).clamp(0, 1),
                    val_poses.cpu(),
                    samples.cpu(),
                    (val_gts.cpu() / 2 + 0.5).clamp(0, 1),
                ], dim=-1)
            else:
                save_obj = torch.cat([
                    (ref_images.cpu() / 2 + 0.5).clamp(0, 1),
                    val_poses.cpu(),
                    samples.cpu(),
                ], dim=-1)
            os.makedirs(f"{output_dir}", exist_ok=True)
            sample_save_path = f"{output_dir}/{data_id}"
            torchvision.utils.save_image(samples.cpu(), sample_save_path, nrow=4)


    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--sanity-check-during-validation", action="store_true")
    args = parser.parse_args()

    exp_config = OmegaConf.load(args.config)
    exp_config["output_dir"] = args.output
    exp_config["unet_checkpoint_path"] = args.ckpt

    main(launcher=args.launcher, sanity_check_during_validation=args.sanity_check_during_validation,
         **exp_config)
