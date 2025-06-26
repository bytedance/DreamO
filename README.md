# DreamO

Official implementation of **[DreamO: A Unified Framework for Image Customization](https://arxiv.org/abs/2504.16915)**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.16915) [![demo](https://img.shields.io/badge/🤗-HuggingFace_Demo-orange)](https://huggingface.co/spaces/ByteDance/DreamO) <br>

### :triangular_flag_on_post: Updates
* **2025.06.26**: [Nunchaku](https://github.com/mit-han-lab/nunchaku) is now supported for model quantization.
* **2025.06.24**: 🔥🔥**We are excited to release DreamO v1.1 with significant improvements in image quality, reduced likelihood of body composition errors, and enhanced aesthetics**. [Learn more about the model](dreamo_v1.1.md)
* **2025.05.30**: 🔥 Native [ComfyUI implementation](https://github.com/ToTheBeginning/ComfyUI-DreamO) is now available!
* **2025.05.12**: Support consumer-grade GPUs (16GB or 24GB) now, see [here](#for-consumer-grade-gpus) for instruction
* **2025.05.11**: We have updated the model to mitigate over-saturation and plastic-face issue. The new version shows consistent improvements over the previous release. Please check it out!
* **2025.05.08**: release codes and models
* 2025.04.24: release DreamO tech report.

https://github.com/user-attachments/assets/385ba166-79df-40d3-bcd7-5472940fa24a

## :wrench: Dependencies and Installation
**note for v1.1**: In order to use Nunchaku for model quantization, we have updated the diffusers version to 0.33.1. If you have the older version 0.31.0 installed, please update diffusers; otherwise, the code will throw errors.
```bash
# clone DreamO repo
git clone https://github.com/bytedance/DreamO.git
cd DreamO
# create conda env
conda create --name dreamo python=3.10
# activate env
conda activate dreamo
# install dependent packages
pip install -r requirements.txt
```
**(optional) Nunchaku**: If you want to use Nunchaku for model quantization, please refer to the [original repo](https://github.com/mit-han-lab/nunchaku) for installation guide.


## :zap: Quick Inference
### Local Gradio Demo
```bash
python app.py
```
```console
options:
  --version {v1.1,v1}   default will use the latest v1.1 model, you can also switch back to v1
  --offload             Enable 'quant=nunchaku' and 'offload' to reduce the original 24GB VRAM to 6.5GB.
  --no_turbo            Use turbo to reduce the original 25 steps to 12 steps.
  --quant {none,int8,nunchaku}
                        Quantize to use: none(bf16), int8, nunchaku
  --device DEVICE       Device to use: auto, cuda, mps, or cpu
```

We observe strong compatibility between DreamO and the accelerated FLUX LoRA variant 
([FLUX-turbo](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha)), and thus enable Turbo LoRA by default, 
reducing inference to 12 steps (vs. 25+ by default). Turbo can be disabled via `--no_turbo`, though our evaluation shows mixed results; 
we therefore recommend keeping Turbo enabled.

**tips**: If you observe limb distortion or poor text generation, try increasing the guidance scale; if the image appears overly glossy or over-saturated, consider lowering the guidance scale.

#### For consumer-grade GPUs
Currently, the code supports two quantization schemes: int8 from [optimum-quanto](https://github.com/huggingface/optimum-quanto) and [Nunchaku](https://github.com/mit-han-lab/nunchaku). You can choose either one based on your needs and the actual results.
- **For users with 8GB GPUs**, run `python app.py --nunchaku --offload` to enable CPU offloading alongside nunchaku quantization. According to the [feedback](https://github.com/bytedance/DreamO/pull/99), it takes about 20 seconds to generate a 1024-resolution image on NVIDIA 3080.

- **For users with 24GB GPUs**, run `python app.py --quant int8` to enable the int8-quantized model or `python app.py --quant nunchaku` to enable the nunchaku-quantized model.

- **For users with 16GB GPUs**, run `python app.py --int8 --offload` to enable CPU offloading alongside int8 quantization. Note that CPU offload significantly reduces inference speed and should only be enabled when necessary.

#### For macOS Apple Silicon (M1/M2/M3/M4)
DreamO now supports macOS with Apple Silicon chips using Metal Performance Shaders (MPS). The app automatically detects and uses MPS when available.

- **For macOS users**, simply run `python app.py` and the app will automatically use MPS acceleration.
- **Manual device selection**: You can explicitly specify the device using `python app.py --device mps` (or `--device cpu` if needed).
- **Memory optimization**: For devices with limited memory, you can combine MPS with quantization: `python app.py --device mps --int8`

**Note**: Make sure you have PyTorch with MPS support installed. The current requirements.txt includes PyTorch 2.6.0+ which has full MPS support.

### Supported Tasks
#### IP
This task is similar to IP-Adapter and supports a wide range of inputs including characters, objects, and animals. 
By leveraging VAE-based feature encoding, DreamO achieves higher fidelity than previous adapter methods, with a distinct advantage in preserving character identity.

![IP_example](https://github.com/user-attachments/assets/086ceabd-338b-4fef-ad1f-bab6b30a1160)

#### ID
Here, ID specifically refers to facial identity. Unlike the IP task, which considers both face and clothing, 
the ID task focuses solely on facial features. This task is similar to InstantID and PuLID. 
Compared to previous methods, DreamO achieves higher facial fidelity, but introduces more model contamination than the SOTA approach PuLID.

![ID_example](https://github.com/user-attachments/assets/392dd325-d4f4-4abb-9718-4b16fe7844c6)

tips: If you notice the face appears overly glossy, try lowering the guidance scale.

#### Try-On
This task supports inputs such as tops, bottoms, glasses, and hats, and enables virtual try-on with multiple garments. 
Notably, our training set does not include multi-garment or ID+garment data, yet the model generalizes well to these unseen combinations.

![tryon_example](https://github.com/user-attachments/assets/fefec673-110a-44f2-83a9-5b779728a734)

#### Style
This task is similar to Style-Adapter and InstantStyle. Please note that style consistency is currently less stable compared to other tasks, 
and in the current version, style cannot be combined with other conditions. We are working on improvements in future releases—stay tuned.

![style_example](https://github.com/user-attachments/assets/0a31674a-c3c2-451f-91e4-c521659d40f3)

#### Multi Condition
You can use multiple conditions (ID, IP, Try-On) to generate more creative images. 
Thanks to the feature routing constraint proposed in the paper, DreamO effectively mitigates conflicts and entanglement among multiple entities.

![multi_cond_example](https://github.com/user-attachments/assets/e43e6ebb-a028-4b29-b76d-3eaa1e69b9c9)

### ComfyUI
- native ComfyUI support: [ComfyUI-DreamO](https://github.com/ToTheBeginning/ComfyUI-DreamO)


### Online HuggingFace Demo
You can try DreamO demo on [HuggingFace](https://huggingface.co/spaces/ByteDance/DreamO).


## Disclaimer

This project strives to impact the domain of AI-driven image generation positively. Users are granted the freedom to
create images using this tool, but they are expected to comply with local laws and utilize it responsibly.
The developers do not assume any responsibility for potential misuse by users.


##  Citation

If DreamO is helpful, please help to ⭐ the repo.

If you find this project useful for your research, please consider citing our [paper](https://arxiv.org/abs/2504.16915).

## :e-mail: Contact
If you have any comments or questions, please [open a new issue](https://github.com/xxx/xxx/issues/new/choose) or contact [Yanze Wu](https://tothebeginning.github.io/) and [Chong Mou](mailto:eechongm@gmail.com).