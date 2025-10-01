# [Real-ESRGAN-AnimeVideo: Upscale anime images and videos](https://aihub.qualcomm.com/models/real_esrgan_animevideo)

Real-ESRGAN AnimeVideo is a machine learning model optimized for upscaling anime content with minimal loss in quality. This model supports both 2x and 4x upscaling.

This is based on the implementation of Real-ESRGAN-AnimeVideo found [here](https://github.com/quangnguyen-ai/Real-ESRGAN). This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices.

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[real-esrgan-animevideo]"
```


Once installed, run the following simple CLI demo:

**For 4x upscaling (default):**
```bash
python -m qai_hub_models.models.real_esrgan_animevideo.demo
```

**For 2x upscaling:**
```bash
python -m qai_hub_models.models.real_esrgan_animevideo.demo --weight-path realesr-animevideox2v3 --scale 2
```

More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

**For 4x upscaling (default):**
```bash
python -m qai_hub_models.models.real_esrgan_animevideo.export --width 640 --height 512 --precision w8a8
```

**For 2x upscaling:**
```bash
python -m qai_hub_models.models.real_esrgan_animevideo.export --weight-path realesr-animevideox2v3 --scale 2 --width 640 --height 512 --precision w8a8
```

Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Real-ESRGAN-AnimeVideo can be found
  [here](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
* [Source Model Implementation](https://github.com/quangnguyen-ai/Real-ESRGAN)
* [Original Implementation](https://github.com/xinntao/Real-ESRGAN/tree/master)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
