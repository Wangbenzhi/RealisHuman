import pdb

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from einops import rearrange
from typing import Union

from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.controlnet import ControlNetConditioningEmbedding, zero_module

from realishuman.models.hack_unet2d import HackUNet2DConditionModel
from realishuman.models.reference_net_attention import ReferenceNetAttention


class RealisHumanUnet(nn.Module):
    def __init__(self, pretrained_model_path,
                 unet_additional_kwargs=None, pose_guider_kwargs=None, clip_projector_kwargs=None, fix_ref_t=True,
                 image_finetune=False, fusion_blocks="full"):
        super(RealisHumanUnet, self).__init__()
        self.image_finetune = image_finetune
        self.fix_ref_t = fix_ref_t

        self.unet_main = HackUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")


        self.unet_ref = HackUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

        self.pose_guider = ControlNetConditioningEmbedding(
                **OmegaConf.to_container(pose_guider_kwargs))

        if clip_projector_kwargs is not None:
            self.clip_projector = nn.Sequential(
                nn.LayerNorm(clip_projector_kwargs.get("in_features")),
                nn.Linear(
                    clip_projector_kwargs.get("in_features"),
                    clip_projector_kwargs.get("out_features"),
                    clip_projector_kwargs.get("bias"),
                    )
            )
        else:
            self.clip_projector = None

        self.reference_writer = ReferenceNetAttention(
            self.unet_ref, mode='write', fusion_blocks=fusion_blocks, is_image=image_finetune)
        self.reference_reader = ReferenceNetAttention(
            self.unet_main, mode='read', fusion_blocks=fusion_blocks, is_image=image_finetune)

    def enable_xformers_memory_efficient_attention(self):
        if is_xformers_available():
            self.unet_ref.enable_xformers_memory_efficient_attention()
            self.unet_main.enable_xformers_memory_efficient_attention()
        else:
            print("xformers is not available, therefore not enabled")

    def enable_gradient_checkpointing(self):
        self.unet_ref.enable_gradient_checkpointing()
        self.unet_main.enable_gradient_checkpointing()

    @property
    def in_channels(self):
        return self.unet_main.config.in_channels

    @property
    def config(self):
        return self.unet_main.config

    @property
    def dtype(self):
        return self.unet_main.dtype

    @property
    def device(self):
        return self.unet_main.device

    def forward(
            self,
            sample: torch.FloatTensor,
            ref_sample: torch.FloatTensor,
            pose: torch.FloatTensor,
            ref_pose: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            drop_reference: bool = False,
            return_dict: bool = True,
    ):

        self.reference_reader.clear()
        self.reference_writer.clear()

        if self.clip_projector is not None:
            encoder_hidden_states = self.clip_projector(encoder_hidden_states)

        if not drop_reference:
            ref_timestep = torch.zeros_like(timestep) if self.fix_ref_t else timestep
            ref_pose_emb = self.pose_guider(ref_pose)
            self.unet_ref(
                ref_sample,
                ref_timestep,
                latent_pose=ref_pose_emb,
                encoder_hidden_states=encoder_hidden_states,  # clip_latents
            )
            self.reference_reader.update(self.reference_writer)

        pose_emb = self.pose_guider(pose)

        model_pred = self.unet_main(
            sample,
            timestep,
            latent_pose=pose_emb,
            encoder_hidden_states=encoder_hidden_states,  # clip_latents
            return_dict=return_dict
        )

        self.reference_reader.clear()
        self.reference_writer.clear()

        return model_pred

    def set_trainable_parameters(self, trainable_modules):
        self.requires_grad_(False)
        for param_name, param in self.named_parameters():
            for trainable_module_name in trainable_modules:
                if trainable_module_name in param_name:
                    param.requires_grad = True
                    break

