# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import sys

import torch

from qai_hub_models.models._shared.super_resolution.model import SuperResolutionModel
from qai_hub_models.utils.asset_loaders import SourceAsRoot
import subprocess 
REALESRGAN_SOURCE_REPOSITORY = "https://github.com/quangnguyen-ai/Real-ESRGAN"


def _get_latest_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "ls-remote", REALESRGAN_SOURCE_REPOSITORY, "HEAD"],
            text=True,
        ).split()[0]
    except Exception as e:
        print(f"Warning: git ls-remote failed ({e}), using pinned commit")
        return "2e2ead76718b03f013314514635576488a2721f6"



REALESRGAN_SOURCE_REPO_COMMIT = _get_latest_commit()
REALESRGAN_SOURCE_VERSION = 1
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "realesr-animevideov3"
PRE_PAD = 10
SCALING_FACTOR = 4


class Real_ESRGAN_AnimeVideo(SuperResolutionModel):
    """Exportable RealESRGAN anime video upscaler, end-to-end."""

    def __init__(
        self,
        realesrgan_model: torch.nn.Module,
        scale_factor: int = SCALING_FACTOR,
    ) -> None:
        super().__init__(realesrgan_model, scale_factor=scale_factor)

    @classmethod
    def from_pretrained(
        cls,
        weight_path: str = DEFAULT_WEIGHTS,
        scale: int = 4,
    ) -> Real_ESRGAN_AnimeVideo:
        """Load Real_ESRGAN_AnimeVideo from a weightfile created by the source RealESRGAN repository.

        Args:
            weight_path: Path to weight file or weight name
            scale: Upscaling factor (2 or 4), default is 4
        """

        # Load PyTorch model from disk
        realesrgan_model = _load_realesrgan_source_model_from_weights(weight_path, scale)

        return cls(realesrgan_model, scale_factor=scale)


def _get_weightsfile_from_name(weights_name: str = DEFAULT_WEIGHTS, scale: int = 4):
    """Convert from names of weights files to the url for the weights file"""
    weights_map = {
        "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "realesr-animevideox2v3": "https://github.com/quangnguyen-ai/Real-ESRGAN/raw/master/weights/realesr-animevideox2v3.pth",
        "smallrealesr-animevideox2v3" :"https://github.com/quangnguyen-ai/Real-ESRGAN/raw/refs/heads/master/weights/small-realesr-animevideox2v3.pth",
        "nanorealesr-animevideox2v3" :"https://github.com/quangnguyen-ai/Real-ESRGAN/raw/refs/heads/master/weights/nano-realesr-animevideox2v3.pth",
        "mediumrealesr-animevideox2v3" :"https://github.com/quangnguyen-ai/Real-ESRGAN/raw/refs/heads/master/weights/medium-realesr-animevideox2v3.pth",
        "large-grayrealesrx2" :"https://github.com/quangnguyen-ai/Real-ESRGAN/raw/refs/heads/master/weights/large-grayrealesrx2.pth",
        "small-grayrealesrx2" :"https://github.com/quangnguyen-ai/Real-ESRGAN/raw/refs/heads/master/weights/small-grayrealesrx2.pth",
        "tiny-grayrealesrx2" :"https://github.com/quangnguyen-ai/Real-ESRGAN/raw/refs/heads/master/weights/tiny-grayrealesrx2.pth",
        "nano-grayrealesrx2" :"https://github.com/quangnguyen-ai/Real-ESRGAN/raw/refs/heads/master/weights/nano-grayrealesrx2.pth",
    }

    if weights_name in weights_map:
        return weights_map[weights_name]

    # If not found in map, return empty string (caller will check if file exists locally)
    return ""


def _load_realesrgan_source_model_from_weights(
    weights_name_or_path: str,
    scale: int = 4,
) -> torch.nn.Module:
    with SourceAsRoot(
        REALESRGAN_SOURCE_REPOSITORY,
        REALESRGAN_SOURCE_REPO_COMMIT,
        MODEL_ID,
        REALESRGAN_SOURCE_VERSION,
    ):
        # Patch path for this load only, since the model source
        # code references modules via a global scope.
        # CWD should be the repository path now
        realesrgan_repo_path = os.getcwd()
        
        # The official repo omits this folder, which causes import issues
        version_dir = os.path.join(realesrgan_repo_path, "realesrgan", "version")
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)

        if os.path.exists(os.path.expanduser(weights_name_or_path)):
            weights_path = os.path.expanduser(weights_name_or_path)
        else:
            weights_path = os.path.join(os.getcwd(), weights_name_or_path + ".pth")
            if not os.path.exists(weights_path):
                # Load RealESRGAN model from the source repository using the given weights.
                # Returns <source repository>.realesrgan.archs.srvgg_arch
                weights_url = _get_weightsfile_from_name(weights_name_or_path, scale)

                # download the weights file
                import requests

                response = requests.get(weights_url)
                with open(weights_path, "wb") as file:
                    file.write(response.content)
                print(f"Weights file downloaded as {weights_path}")

        # necessary import. `archs` comes from the realesrgan repo.
        # This can be imported only once per session
        if "basicsr.archs.srvgg_arch" not in sys.modules:
            # -----
            # Patch torchvision for out of date basicsr package that requires torchvision 1.16
            # but does not have its requirements set correctly
            try:
                # This is not available after torchvision 1.16, it was renamed to "functional"
                import torchvision.transforms.functional_tensor
            except ImportError:
                import torchvision.transforms.functional

                sys.modules["torchvision.transforms.functional_tensor"] = (
                    torchvision.transforms.functional
                )
            # ----
            import realesrgan.archs.srvgg_arch as srvgg_arch
        else:
            srvgg_arch = sys.modules["basicsr.archs.srvgg_arch"]

        # Anime video models use XS size (num_conv=16) instead of S size (num_conv=32)
        if "gray" in weights_path: 
            
            if "large" in weights_path:
                realesrgan_model = srvgg_arch.SRVGGNetMobileInfer(
                    num_in_ch=1,
                    num_out_ch=1,
                    num_conv64feat=4, 
                    num_conv32feat=32,
                    upscale=scale,
                    act_type="prelu",
                    use_skip=True,
                )    
                
            elif "small" in weights_path:
                realesrgan_model = srvgg_arch.SRVGGNetMobileInfer(
                    num_in_ch=1,
                    num_out_ch=1,
                    num_conv64feat=2, 
                    num_conv32feat=8,
                    upscale=scale,
                    act_type="relu",
                    use_skip=True,
                )    
            elif "tiny" in weights_path: 
                realesrgan_model = srvgg_arch.SRVGGNetMobileInfer(
                    num_in_ch=1,
                    num_out_ch=1,
                    num_conv64feat=0,
                    num_conv32feat=8,
                    upscale=scale,
                    act_type="relu",
                    use_skip= False
                )
            else: ##nano
                realesrgan_model = srvgg_arch.SRVGGNetMobileInfer(
                    num_in_ch=1,
                    num_out_ch=1,
                    num_conv64feat=0,
                    num_conv32feat=3,
                    upscale=scale,
                    act_type="relu",
                    use_skip= False
                )
                
        
        else:
            if "nano" in weights_path:
                realesrgan_model = srvgg_arch.SRVGGNetCompactInfer(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=32,
                    num_conv=16,  # XS size for anime video
                    upscale=scale,
                    act_type="prelu",
                )
            elif "medium" in weights_path:
                realesrgan_model = srvgg_arch.SRVGGNetCompactInfer(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=48,
                    num_conv=16,  # XS size for anime video
                    upscale=scale,
                    act_type="prelu",
                )
                
            elif "small" in weights_path:
                realesrgan_model = srvgg_arch.SRVGGNetCompactInfer(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=32,
                    num_conv=32,  # XS size for anime video
                    upscale=scale,
                    act_type="prelu",
                )
            else: 
                realesrgan_model = srvgg_arch.SRVGGNetCompactInfer(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_conv=16,  # XS size for anime video
                    upscale=scale,
                    act_type="prelu",
                )
        pretrained_dict = torch.load(weights_path, map_location=torch.device("cpu"))

        if "params_ema" in pretrained_dict:
            keyname = "params_ema"
        else:
            keyname = "params"
        realesrgan_model.load_state_dict(pretrained_dict[keyname], strict=True)

        return realesrgan_model