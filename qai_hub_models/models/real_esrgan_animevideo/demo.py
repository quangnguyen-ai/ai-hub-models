# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.super_resolution.demo import super_resolution_demo
from qai_hub_models.models.real_esrgan_animevideo.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Real_ESRGAN_AnimeVideo,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

WEIGHTS_HELP_MSG = "RealESRGAN checkpoint `.pth` name from the Real-ESRGAN repo. Can be set to any of the model names defined here: https://github.com/quangnguyen-ai/Real-ESRGAN to automatically download the file instead."
# Use demo image from real_esrgan_general_x4v3 model
IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    "real_esrgan_general_x4v3", 2, "real_esrgan_general_x4v3_demo.jpg"
)


# Run Real-ESRGAN AnimeVideo end-to-end on a sample image.
# The demo will display an image with upscaled output.
def main(is_test: bool = False):
    super_resolution_demo(
        model_cls=Real_ESRGAN_AnimeVideo,
        model_id=MODEL_ID,
        default_image=IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
