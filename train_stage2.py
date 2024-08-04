import os
import math
import logging
import inspect
import argparse
import datetime
import random
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import compute_snr

from realishuman.models.realishuman_unet_paste_inpaint import PasteInpaintHandUnet
from realishuman.utils.util import get_distributed_dataloader, sanity_check


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

    name: str,
    launcher: str,

    output_dir: str,
    pretrained_model_path: str,
    # pretrained_clip_path: str,

    train_data: Dict,
    train_cfg: bool = True,
    cfg_uncond_ratio: float = 0.1,

    pretrained_vae_path: str = "",
    hand_prompt: str = "Natural inpainting the area around the hand",
    unet_checkpoint_path: str = "",
    zero_snr: bool = False,
    snr_gamma: Optional[float] = None,
    v_pred: bool = False,

    max_train_epoch: int = -1,
    max_train_steps: int = 100,

    learning_rate: float = 1e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None,),
    num_workers: int = 4,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,
    checkpointing_steps_tuple: Tuple[int] = (),

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,

    *args,
    **kwargs,
):
    # check version
    check_min_version("0.25.0")

    # Initialize distributed training
    local_rank = init_dist(launcher=launcher)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0
    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")
    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))

    if train_cfg and is_main_process:
        logging.info(f"Enable CFG training with drop rate {cfg_uncond_ratio}.")

    # Load scheduler, tokenizer and models
    if is_main_process:
        logging.info("Load scheduler, tokenizer and models.")
    if pretrained_vae_path != "":
        vae = AutoencoderKL.from_pretrained(pretrained_vae_path, subfolder="sd-vae-ft-mse")
    else:
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    # image_encoder = AutoModel.from_pretrained(pretrained_clip_path)

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

    unet = PasteInpaintHandUnet(
        pretrained_model_path=pretrained_model_path,
        image_finetune=image_finetune,
        # unet_additional_kwargs=unet_additional_kwargs,
    )

    # Load pretrained unet weights


    if unet_checkpoint_path != "":
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

    # Set unet trainable parameters
    unet.set_trainable_parameters(trainable_modules)

    # Set learning rate and optimizer
    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    trainable_parameter_keys = []
    trainable_params = []
    for param_name, param in unet.named_parameters():
        if param.requires_grad:
            trainable_parameter_keys.append(param_name)
            trainable_params.append(param)
    if is_main_process:
        logging.info(f"trainable params number: {trainable_parameter_keys}")
        logging.info(f"trainable params number: {len(trainable_params)}")
        logging.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Set learning rate scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Freeze vae and image_encoder
    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # move to cuda
    vae.to(local_rank)
    text_encoder.to(local_rank)
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Get the training dataloader
    train_dataloader = get_distributed_dataloader(
        dataset_config=train_data,
        batch_size=train_batch_size,
        num_processes=num_processes,
        num_workers=num_workers,
        shuffle=True,
        global_rank=global_rank,
        seed=global_seed,)

    # Get the training iteration
    overrode_max_train_steps = False
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        overrode_max_train_steps = True

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = max_train_epoch * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataloader)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None
    with torch.no_grad():
        prompt_ids = tokenizer(
            [hand_prompt]*train_batch_size, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        ).input_ids.to(local_rank)
        encoder_hidden_states = text_encoder(prompt_ids)[0]
        uncond_prompt_ids = tokenizer(
            [""]*train_batch_size, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        ).input_ids.to(local_rank)
        ucond_encoder_hidden_states = text_encoder(uncond_prompt_ids)[0]
    

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                sanity_check(batch, f"{output_dir}/sanity_check", image_finetune, global_rank)

            """ >>>> Training >>>> """
            # Get images
            image = batch["image"].to(local_rank) #[b, c, f, h, w]
            masked_image = batch["bg_image"].to(local_rank) #masked image -> masked latents
            mask = batch["mask"].to(local_rank) 

            if train_cfg and random.random() < cfg_uncond_ratio:
                encoder_hidden_states_input = ucond_encoder_hidden_states
                drop_reference = True
            else:
                encoder_hidden_states_input = encoder_hidden_states
                drop_reference = False

            # Convert images to latent space
            with torch.no_grad():

                latents = vae.encode(image).latent_dist
                latents = latents.sample()
                latents = latents * vae.config.scaling_factor

                masked_latents = vae.encode(masked_image).latent_dist
                masked_latents = masked_latents.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
     
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(train_data.args.sample_size[0] // 8, train_data.args.sample_size[0] // 8))

                # Sample noise that we"ll add to the latents
                bsz = latents.shape[0]
                noise = torch.randn_like(latents)

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):

                model_pred = unet(
                    sample=latent_model_input, #noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states_input,
                ).sample

                if snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    snr = snr.clamp(min=1e-8)  # incase zero-snr
                    mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1
            """ <<<< Training <<<< """

            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or global_step in checkpointing_steps_tuple):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unet.state_dict(),
                }
                torch.save(state_dict, os.path.join(save_path, f"checkpoint-iter-{global_step}.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if is_main_process and global_step % 500 == 0:
                logging.info(f"step: {global_step} / {max_train_steps}:  {logs}")
            if global_step >= max_train_steps:
                break

    # save the final checkpoint
    if is_main_process:
        save_path = os.path.join(output_dir, f"checkpoints")
        state_dict = {
            "epoch": num_train_epochs - 1,
            "global_step": global_step,
            "state_dict": unet.state_dict(),
        }
        torch.save(state_dict, os.path.join(save_path, f"checkpoint-final.ckpt"))
        logging.info(f"Saved final state to {save_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    args = parser.parse_args()

    exp_name = Path(args.config).stem
    exp_config = OmegaConf.load(args.config)

    main(name=exp_name, launcher=args.launcher, **exp_config)