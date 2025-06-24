# DreamO v1.1

After the release of DreamO v1, we received valuable feedback from the community. We're grateful for your support and suggestions.

The main issues identified were:
1. Overly glossy or plastic-like rendering.
2. High probability of structural or anatomical artifacts.

In version 1.1, we addressed these through post-training with SFT and DPO on high-quality datasets. Key improvements and comparisons are shown below.

### Structure & Anatomy

In stylized scenes, v1 frequently suffered from structural and anatomical failures. v1.1 significantly improves stability, as illustrated:

![Image](https://github.com/user-attachments/assets/83c845f5-72ce-4990-93f9-c14301fba361)

In realistic scenes, v1.1 offers better hand and body rendering, with improved overall composition and aesthetics:

![Image](https://github.com/user-attachments/assets/0fa7fdd3-9497-4710-8b36-382aae9be144)

### Glossy/Plastic Artifacts

v1.1 reduces the excessive glossiness and plastic-like appearance in faces and scenes, resulting in more natural and appealing outputs:

![Image](https://github.com/user-attachments/assets/cc623a37-6d0a-4f74-993a-b6a5a4bff7ea)

![Image](https://github.com/user-attachments/assets/b540c483-ab1f-4ede-9a0f-3f774c1dd358)

## Tips

- Applying the FLUX LoRA [Super-Realism](https://huggingface.co/strangerzonehf/Flux-Super-Realism-LoRA) can further enhance realism. However, it may interfere with stylization, so it's not included by default. You can experiment with it in ComfyUI. If you discover better LoRAs, we welcome contributions.
  
- By default, reference images are resized to 512×512. For images with fine text or intricate details, consider increasing the resolution. This enables the generated image to retain more details of the reference image, see the figure below. However, only do so when necessary, as it increases inference time and may reduce editability.

  ![Image](https://github.com/user-attachments/assets/d00fb760-1def-4000-8f64-7570790404b3)

## How to Update & Use
- Diffuser: Pull the latest code from this repo and run `app.py` as before. v1.1 is used by default and will automatically download from [Hugging Face](https://huggingface.co/ByteDance/DreamO/tree/main/v1.1).
- ComfyUI: Refer to our official native implementation here: [ComfyUI-DreamO](https://github.com/ToTheBeginning/ComfyUI-DreamO).
- Hugging Face Demo: The online demo has been updated to use the latest v1.1 model.


## Limitation

Compared to v1, v1.1 may perform slightly worse in distinguishing multiple similar subjects within the same composition.
