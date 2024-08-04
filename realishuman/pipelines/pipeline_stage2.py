# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.utils import is_accelerate_available
from packaging import version

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from realishuman.models.realishuman_unet_paste_inpaint import PasteInpaintHandUnet
from realishuman.pipelines.context import (
    get_context_scheduler,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class StageTwoPipelineOutput(BaseOutput):
    sample: Optional[Union[torch.Tensor, np.ndarray]] = None


class StageTwoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: Union[AutoencoderKL, AutoencoderKLTemporalDecoder],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: PasteInpaintHandUnet,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ]
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, uncond_prompt, device):
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length, padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            encoder_hidden_states = self.text_encoder(prompt_ids)[0]
            uncond_prompt_ids = self.tokenizer(
                uncond_prompt, max_length=self.tokenizer.model_max_length, padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            ucond_encoder_hidden_states = self.text_encoder(uncond_prompt_ids)[0]

        return encoder_hidden_states, ucond_encoder_hidden_states

    def decode_latents(self, latents, decode_chunk_size=14):
        latents = 1 / self.vae.config.scaling_factor * latents
        # decode decode_chunk_size frames at a time to avoid OOM
        sample = []
        for frame_idx in range(0, latents.shape[0], decode_chunk_size):
            sample.append(self.vae.decode(latents[frame_idx:frame_idx+decode_chunk_size]).sample)
        sample = torch.cat(sample)
        sample = (sample / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        sample = sample.cpu().float().numpy()
        return sample

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")


        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
    def prepare_mask_latents(
        self, mask, masked_image, mask_finalstep, masked_image_finalstep, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):         
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(1), size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)
        
        masked_image = masked_image.to(device=device, dtype=dtype)
        masked_image_latents = self.vae.encode(masked_image).latent_dist
        masked_image_latents = masked_image_latents.sample()
        masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        
        
        mask_finalstep = torch.nn.functional.interpolate(
            mask_finalstep.unsqueeze(1), size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask_finalstep = mask_finalstep.to(device=device, dtype=dtype)
        
        masked_image_finalstep = masked_image_finalstep.to(device=device, dtype=dtype)
        masked_image_finalstep_latents = self.vae.encode(masked_image_finalstep).latent_dist
        masked_image_finalstep_latents = masked_image_finalstep_latents.sample()
        masked_image_finalstep_latents = masked_image_finalstep_latents * self.vae.config.scaling_factor
     
        
        # masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
            mask_finalstep = mask_finalstep.repeat(batch_size // mask_finalstep.shape[0], 1, 1, 1)
            
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)
            masked_image_finalstep_latents = masked_image_finalstep_latents.repeat(batch_size // masked_image_finalstep_latents.shape[0], 1, 1, 1)
            
        # mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        # masked_image_latents = (
        #     torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        # )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        masked_image_finalstep_latents = masked_image_finalstep_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents, mask_finalstep, masked_image_finalstep_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device,
                        generator, latents=None):

        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width //
                 self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).contiguous().to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: torch.FloatTensor,
        bg_image: torch.FloatTensor,
        mask: torch.FloatTensor,
        mask_finalstep: torch.FloatTensor,
        bg_image_finalstep: torch.FloatTensor,
        hand_prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        finalstep_number: Optional[int] = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        **kwargs,
    ):
        # TODO: support multiple images per prompt
        assert num_images_per_prompt == 1, "not support multiple images per prompt yet"

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(hand_prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = bg_image.shape[0]
        if isinstance(hand_prompt, list):
            assert len(hand_prompt) == batch_size
        else:
            hand_prompt = [hand_prompt] * batch_size

        device = self._execution_device

        uncond_prompt = ["" for _ in hand_prompt]
        # Encode input prompt
        clip_latents, uncond_clip_latents = self._encode_prompt(hand_prompt, uncond_prompt, device)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            clip_latents.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype
        
        gt_latents = self.vae.encode(image).latent_dist
        gt_latents = gt_latents.sample()
        gt_latents = gt_latents * self.vae.config.scaling_factor
        
        # ref_mean_ratio = 0.1
        # latents = gt_latents * ref_mean_ratio + (1 - ref_mean_ratio) * latents

        mask, masked_image_latents, mask_finalstep, masked_image_finalstep_latents = self.prepare_mask_latents(
            mask,
            bg_image,
            mask_finalstep,
            bg_image_finalstep,
            batch_size * num_images_per_prompt,
            height,
            width,
            latents_dtype,
            device,
            generator,
            guidance_scale>0,
        )


        num_channels_unet = self.unet.config.in_channels
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise = torch.randn_like(latents)
                noisy_gt = self.scheduler.add_noise(gt_latents, noise, t)
                #mask_finalstep: 1:foreground, 0:backgroud
                latents = latents * mask_finalstep + noisy_gt * (1 - mask_finalstep)
                latent_model_input = latents 
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=clip_latents,
                ).sample.to(dtype=latents_dtype)
                if guidance_scale > 1.0:
                    noise_uncond = self.unet(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=uncond_clip_latents, 
                    ).sample.to(dtype=latents_dtype)
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        latents = latents * mask_finalstep + gt_latents * (1 - mask_finalstep)
        # Post-processing
        sample = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            sample = torch.from_numpy(sample)

        if not return_dict:
            return sample

        return StageTwoPipelineOutput(sample=sample)
