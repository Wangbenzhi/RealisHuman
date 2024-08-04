import pdb

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from einops import rearrange
from typing import Union

from diffusers.utils.import_utils import is_xformers_available
from realishuman.models.hack_unet2d import HackUNet2DConditionModel
from realishuman.models.reference_net_attention import ReferenceNetAttention


class PasteInpaintHandUnet(nn.Module):
    def __init__(self, pretrained_model_path,
                 unet_additional_kwargs=None, image_finetune=False, use_hp=False, use_sd_vae=False):
        super(PasteInpaintHandUnet, self).__init__()

        self.unet_main = HackUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    def enable_xformers_memory_efficient_attention(self):
        if is_xformers_available():
            self.unet_main.enable_xformers_memory_efficient_attention()
        else:
            print("xformers is not available, therefore not enabled")

    def enable_gradient_checkpointing(self):
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
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            return_dict: bool = True,
    ):


        model_pred = self.unet_main(
            sample,
            timestep,
            latent_pose=None,
            encoder_hidden_states=encoder_hidden_states,  # clip_latents
            return_dict=return_dict,
        )

        return model_pred

    def set_trainable_parameters(self, trainable_modules):
        self.requires_grad_(False)
        for param_name, param in self.named_parameters():
            for trainable_module_name in trainable_modules:
                if trainable_module_name in param_name:
                    param.requires_grad = True
                    break

