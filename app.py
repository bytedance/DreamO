# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download
from optimum.quanto import freeze, qint8, quantize
from PIL import Image
from torchvision.transforms.functional import normalize

from dreamo.dreamo_pipeline import DreamOPipeline
from dreamo.utils import (
    img2tensor,
    resize_numpy_image_area,
    resize_numpy_image_long,
    tensor2img,
)
from tools import BEN2

# Argument parsing with debug flag
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--no_turbo', action='store_true')
parser.add_argument('--int8', action='store_true')
parser.add_argument('--offload', action='store_true')
parser.add_argument('--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

# Debug print function
def dprint(*msg, **kwargs):
    if args.debug:
        print('[DEBUG]', *msg, **kwargs)

def list_loras(lora_dir):
    if not os.path.exists(lora_dir):
        os.makedirs(lora_dir)
    lora_files = [f for f in os.listdir(lora_dir) if f.lower().endswith(('.safetensors', '.pt'))]
    dprint(f"Found {len(lora_files)} LoRA files in '{lora_dir}': {lora_files}")
    return lora_files

class Generator:
    def __init__(self):
        init_start = time.time()
        self.device = torch.device('cuda')
        dprint("Initializing BEN2 background removal model...")
        self.bg_rm_model = BEN2.BEN_Base().to(self.device).eval()
        dprint("Checking for BEN2 model file in 'models' directory...")
        hf_hub_download(repo_id='PramaLLC/BEN2', filename='BEN2_Base.pth', local_dir='models')
        self.bg_rm_model.loadcheckpoints('models/BEN2_Base.pth')
        self.face_helper = None
        self.loaded_loras = None
        dprint("BEN2 ready. (Elapsed: {:.2f}s)".format(time.time()-init_start))

        if args.offload:
            dprint("Moving BEN2 model to CPU (offload enabled).")
            self.ben_to_device(torch.device('cpu'))

        #model_root = 'black-forest-labs/FLUX.1-dev'
        model_root = 'ChuckMcSneed/FLUX.1-dev'
        dprint("Loading DreamOPipeline from pretrained weights...")
        dreamo_pipeline = DreamOPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
        dprint("Loading DreamO model weights...")
        dreamo_pipeline.load_dreamo_model(self.device, use_turbo=not args.no_turbo)
        quantized_model_path = "models/quantized_transformer.pt"
        quantized_encoder_path = "models/quantized_text_encoder_2.pt"
        try:
            if args.int8:
                dprint('int8 flag detected, attempting to load or create quantized models...')
                quantized_loaded = False
                if os.path.exists(quantized_model_path) and os.path.exists(quantized_encoder_path):
                    try:
                        dprint('Trying to load quantized models from disk...')
                        dreamo_pipeline.transformer = torch.load(quantized_model_path, map_location=self.device, weights_only=False)
                        dreamo_pipeline.text_encoder_2 = torch.load(quantized_encoder_path, map_location=self.device, weights_only=False)
                        dprint('Loaded quantized models.')
                        quantized_loaded = True
                    except Exception as e:
                        dprint(f"Failed to load quantized models, will quantize anew: {e}")
                if not quantized_loaded:
                    dprint('Quantizing models...')
                    q_start = time.time()
                    quantize(dreamo_pipeline.transformer, qint8)
                    freeze(dreamo_pipeline.transformer)
                    quantize(dreamo_pipeline.text_encoder_2, qint8)
                    freeze(dreamo_pipeline.text_encoder_2)
                    dprint('Quantization done. (Elapsed: {:.2f}s)'.format(time.time()-q_start))
                    try:
                        dprint('Saving quantized models to disk...')
                        torch.save(dreamo_pipeline.transformer, quantized_model_path)
                        torch.save(dreamo_pipeline.text_encoder_2, quantized_encoder_path)
                        dprint('Saved quantized models.')
                    except Exception as e:
                        dprint(f"Failed to save quantized models: {e}")
            self.dreamo_pipeline = dreamo_pipeline.to(self.device)
            dprint("DreamO pipeline loaded and moved to device.")
        except Exception as e:
            dprint(f"Exception during quantization or pipeline setup: {e}")
            raise

        if args.offload:
            dprint("Enabling model CPU offload for DreamO pipeline...")
            self.dreamo_pipeline.enable_model_cpu_offload()
            self.dreamo_pipeline.offload = True
            dprint("DreamO pipeline CPU offload enabled.")
        else:
            self.dreamo_pipeline.offload = False

        dprint("Generator initialization complete. (Total Elapsed: {:.2f}s)".format(time.time()-init_start))

    def manage_loras(self, selected_loras, lora_dir):
        dprint(f"LoRA management requested. Current: {self.loaded_loras}, New: {selected_loras}")
        if self.loaded_loras == selected_loras:
            dprint("No change in LoRA set; skipping reload.")
            return  # No change, skip reload
        start = time.time()
        dprint("Unloading previous LoRAs (if any)...")
        self.dreamo_pipeline = unload_all_loras(self.dreamo_pipeline)
        dprint("Previous LoRAs unloaded. (Elapsed: {:.2f}s)".format(time.time()-start))
        for lora_name, lora_weight in selected_loras:
            lora_path = os.path.join(lora_dir, lora_name)
            if os.path.exists(lora_path):
                lora_load_start = time.time()
                dprint(f"Loading LoRA '{lora_name}' with weight {lora_weight}...")
                self.dreamo_pipeline.load_lora_weights(lora_path, weight=lora_weight)
                dprint(f"Loaded LoRA '{lora_name}'. (Elapsed: {time.time()-lora_load_start:.2f}s)")
            else:
                dprint(f"LoRA file '{lora_path}' not found; skipping.")
        self.loaded_loras = selected_loras
        dprint("LoRA management done. (Total Elapsed: {:.2f}s)".format(time.time()-start))

    def ben_to_device(self, device):
        dprint(f"Moving BEN2 background removal model to device: {device}")
        self.bg_rm_model.to(device)

    def facexlib_to_device(self, device):
        if self.face_helper is not None:
            dprint(f"Moving FaceRestoreHelper models to device: {device}")
            self.face_helper.face_det.to(device)
            self.face_helper.face_parse.to(device)

    def init_face_helper(self, upscale_factor, face_size, crop_ratio):
        dprint(f"Initializing FaceRestoreHelper (upscale_factor={upscale_factor}, face_size={face_size}, crop_ratio={crop_ratio})")
        self.face_helper = FaceRestoreHelper(
            upscale_factor=int(upscale_factor),
            face_size=int(face_size),
            crop_ratio=(float(crop_ratio), float(crop_ratio)),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )

    @torch.no_grad()
    def get_align_face(self, img, upscale_factor, face_size, crop_ratio):
        align_start = time.time()
        dprint("Starting face alignment...")
        self.init_face_helper(upscale_factor, face_size, crop_ratio)
        self.face_helper.clean_all()
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            dprint("No face detected for alignment.")
            return None
        align_face = self.face_helper.cropped_faces[0]
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        face_features_image = torch.where(bg, white_image, input)
        face_features_image = tensor2img(face_features_image, rgb2bgr=False)
        dprint(f"Face alignment finished. (Elapsed: {time.time()-align_start:.2f}s)")
        return face_features_image

generator = Generator()

@torch.inference_mode()
def unload_all_loras(pipeline):
    try:
        dprint("Attempting to unload all LoRA weights from pipeline...")
        pipeline.unload_lora_weights()
        dprint("Unloaded LoRA weights from pipeline.")
    except AttributeError:
        dprint("Pipeline does not support unload_lora_weights; reloading base model.")
        base_model_path = 'black-forest-labs/FLUX.1-dev'
        pipeline = DreamOPipeline.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
        pipeline.load_dreamo_model(torch.device('cuda'), use_turbo=not args.no_turbo)
    return pipeline

def generate_image(
    ref_image1,
    ref_image2,
    ref_task1,
    ref_task2,
    prompt,
    width,
    height,
    ref_res,
    num_steps,
    guidance,
    seed,
    true_cfg,
    cfg_start_step,
    cfg_end_step,
    neg_prompt,
    neg_guidance,
    first_step_guidance,
    lora_dir,
    lora_1, lora_2, lora_3, lora_4, lora_5,
    weight_1, weight_2, weight_3, weight_4, weight_5,
    face_size,
    upscale_factor,
    crop_ratio,
    num_images  # <-- This will be a string from gr.Textbox
):
    try:
        num_images = int(num_images)
        if num_images < 1:
            num_images = 1
    except Exception:
        num_images = 1
    
    total_start = time.time()
    dprint("=== GENERATION STARTED ===")
    dprint(f"Prompt: {prompt}")
    dprint(f"Parameters: width={width}, height={height}, steps={num_steps}, guidance={guidance}, seed={seed}, num_images={num_images}")

    ref_conds = []
    debug_images = []

    lora_files = [lora_1, lora_2, lora_3, lora_4, lora_5]
    lora_weights = [weight_1, weight_2, weight_3, weight_4, weight_5]
    selected_loras = [(f, w) for f, w in zip(lora_files, lora_weights) if f and f != "None"]
    dprint(f"Selected LoRAs: {selected_loras}")

    manage_start = time.time()
    generator.manage_loras(selected_loras, lora_dir)
    dprint(f"manage_loras() completed in {time.time() - manage_start:.2f}s")

    ref_images = [ref_image1, ref_image2]
    ref_tasks = [ref_task1, ref_task2]

    for idx, (ref_image, ref_task) in enumerate(zip(ref_images, ref_tasks)):
        if ref_image is not None:
            img_start = time.time()
            dprint(f"Processing reference image {idx+1} for task '{ref_task}'...")
            if ref_task == "id":
                if args.offload:
                    dprint("Moving FaceHelper to CUDA for alignment...")
                    generator.facexlib_to_device(torch.device('cuda'))
                ref_image = resize_numpy_image_long(ref_image, 1024)
                ref_image = generator.get_align_face(ref_image, upscale_factor, face_size, crop_ratio)
                if args.offload:
                    dprint("Moving FaceHelper to CPU after alignment...")
                    generator.facexlib_to_device(torch.device('cpu'))
            elif ref_task != "style":
                if args.offload:
                    dprint("Moving BEN2 to CUDA for background removal...")
                    generator.ben_to_device(torch.device('cuda'))
                ref_image = generator.bg_rm_model.inference(Image.fromarray(ref_image))
                if args.offload:
                    dprint("Moving BEN2 to CPU after background removal...")
                    generator.ben_to_device(torch.device('cpu'))
            if ref_task != "id":
                dprint("Resizing reference image to target area...")
                ref_image = resize_numpy_image_area(np.array(ref_image), ref_res * ref_res)
            debug_images.append(ref_image)
            ref_image = img2tensor(ref_image, bgr2rgb=False).unsqueeze(0) / 255.0
            ref_image = 2 * ref_image - 1.0
            ref_conds.append(
                {
                    'img': ref_image,
                    'task': ref_task,
                    'idx': idx + 1,
                }
            )
            dprint(f"Reference image {idx+1} processing complete. (Elapsed: {time.time()-img_start:.2f}s)")

    # Handle seed logic for batch generation
    try:
        seed = int(seed)
    except Exception:
        dprint(f"Invalid seed '{seed}', defaulting to -1.")
        seed = -1

    images = []
    seeds_used = []

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        # For batch reproducibility: use user seed + index if user gave a non-random seed, else random each time
        if seed == -1:
            this_seed = torch.randint(0, 2**63 - 1, (1,)).item()
        else:
            this_seed = (seed + i) % (2**63 - 1)
        dprint(f"Generating image {i+1}/{num_images} with seed {this_seed}...")
        generator_obj = torch.Generator(device="cpu").manual_seed(this_seed)

        inference_start = time.time()
        result = generator.dreamo_pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            ref_conds=ref_conds,
            generator=generator_obj,
            true_cfg_scale=true_cfg,
            true_cfg_start_step=cfg_start_step,
            true_cfg_end_step=cfg_end_step,
            negative_prompt=neg_prompt,
            neg_guidance_scale=neg_guidance,
            first_step_guidance_scale=first_step_guidance if first_step_guidance > 0 else guidance,
        )
        image = result.images[0]
        dprint(f"DreamO pipeline inference complete for image {i+1}. (Elapsed: {time.time() - inference_start:.2f}s)")

        filename = f"output_{this_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        filepath = os.path.join(output_dir, filename)
        save_start = time.time()
        if isinstance(image, Image.Image):
            image.save(filepath, format="PNG")
        else:
            Image.fromarray(image).save(filepath, format="PNG")
        dprint(f"Image {i+1} saved to {filepath}. (Elapsed: {time.time() - save_start:.2f}s)")

        images.append(image)
        seeds_used.append(str(this_seed))

    dprint(f"=== GENERATION COMPLETED === (Total Elapsed: {time.time() - total_start:.2f}s)")
    # For batch, output images as list, debug_images (from last run), and seeds used as joined string
    return images, debug_images, ', '.join(seeds_used)

_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">DreamO</h1>
    <p style="font-size: 1rem; margin-bottom: 1.5rem;">Paper: <a href='https://arxiv.org/abs/2504.16915' target='_blank'>DreamO: A Unified Framework for Image Customization</a> | Codes: <a href='https://github.com/bytedance/DreamO' target='_blank'>GitHub</a></p>
</div>
🚩 Update Notes:
- 2025.05.11: We have updated the model to mitigate over-saturation and plastic-face issues. The new version shows consistent improvements over the previous release.
❗️❗️❗️**User Guide:**
- The most important thing to do first is to try the examples provided below the demo, which will help you better understand the capabilities of the DreamO model and the types of tasks it currently supports
- For each input, please select the appropriate task type. For general objects, characters, or clothing, choose IP — we will remove the background from the input image. If you select ID, we will extract the face region from the input image (similar to PuLID). If you select Style, the background will be preserved, and you must prepend the prompt with the instruction: 'generate a same style image.' to activate the style task.
- The most import hyperparameter in this demo is the guidance scale, which is set to 3.5 by default. If you notice that faces appear overly glossy or unrealistic—especially in ID tasks—you can lower the guidance scale (e.g., to 3). Conversely, if text rendering is poor or limb distortion occurs, increasing the guidance scale (e.g., to 4) may help.
- To accelerate inference, we adopt FLUX-turbo LoRA, which reduces the sampling steps from 25 to 12 compared to FLUX-dev. Additionally, we distill a CFG LoRA, achieving nearly a twofold reduction in steps by eliminating the need for true CFG
'''

_CITE_ = r"""
If DreamO is helpful, please help to ⭐ the <a href='https://github.com/bytedance/DreamO' target='_blank'> Github Repo</a>. Thanks!
---
📧 **Contact**
If you have any questions or feedbacks, feel free to open a discussion or contact <b>wuyanze123@gmail.com</b> and <b>eechongm@gmail.com</b>
"""

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown(_HEADER_)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    ref_image1 = gr.Image(label="ref image 1", type="numpy", height=256)
                    ref_image2 = gr.Image(label="ref image 2", type="numpy", height=256)
                with gr.Row():
                    ref_task1 = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="task for ref image 1")
                    ref_task2 = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="task for ref image 2")
                prompt = gr.Textbox(label="Prompt", value="a person playing guitar in the street")
                width = gr.Slider(768, 4096, 1024, step=16, label="Width")
                height = gr.Slider(768, 4096, 1024, step=16, label="Height")
                num_steps = gr.Slider(8, 100, 12, step=1, label="Number of steps")
                guidance = gr.Slider(1.0, 50.0, 3.5, step=0.1, label="Guidance")
                seed = gr.Textbox(label="Seed (-1 for random)", value="-1")
                num_images = gr.Textbox(value="1", label="Number of Images to Generate (integer)")
                with gr.Accordion("Advanced Options", open=False, visible=True):
                    ref_res = gr.Slider(512, 2048, 512, step=16, label="resolution for ref image")
                    neg_prompt = gr.Textbox(label="Neg Prompt", value="")
                    neg_guidance = gr.Slider(1.0, 10.0, 3.5, step=0.1, label="Neg Guidance")
                    true_cfg = gr.Slider(1, 5, 1, step=0.1, label="true cfg")
                    cfg_start_step = gr.Slider(0, 30, 0, step=1, label="cfg start step")
                    cfg_end_step = gr.Slider(0, 30, 0, step=1, label="cfg end step")
                    first_step_guidance = gr.Slider(0, 10, 0, step=0.1, label="first step guidance")
                    face_size = gr.Slider(512, 1024, 512, step=512, label="Face Crop Size (pixels)")
                    upscale_factor = gr.Dropdown(choices=[1, 2], value=2, label="Face Upscale Factor")
                    crop_ratio = gr.Slider(1.0, 1.3, 1.0, step=0.05, label="Face Crop Ratio (advanced)", visible=True)

                    # --- LoRA Controls here ---
                    lora_dir = gr.Textbox(value="loras", label="LoRA directory")
                    refresh_lora_btn = gr.Button("Refresh LoRAs")

                    lora_dropdowns = []
                    lora_weights = []
                    for i in range(5):
                        lora_dropdown = gr.Dropdown(choices=["None"], label=f"LoRA {i+1}", value="None")
                        lora_weight = gr.Slider(-5, 5, 1, step=0.01, label=f"Weight {i+1}")
                        lora_dropdowns.append(lora_dropdown)
                        lora_weights.append(lora_weight)

                    def refresh_lora_list(lora_dir):
                        lora_files = list_loras(lora_dir)
                        choices = ["None"] + lora_files
                        return [gr.update(choices=choices, value="None") for _ in range(5)]

                    refresh_lora_btn.click(
                        refresh_lora_list,
                        inputs=[lora_dir],
                        outputs=lora_dropdowns,
                    )

                generate_btn = gr.Button("Generate")
                gr.Markdown(_CITE_)

            with gr.Column():
                output_gallery = gr.Gallery(label="Generated Images", show_label=True, elem_id="gallery", columns=2, height="auto")
                debug_image = gr.Gallery(
                    label="Preprocessing output (including possible face crop and background remove)",
                    elem_id="debug_gallery",
                )
                seed_output = gr.Textbox(label="Used Seeds")

        with gr.Row(), gr.Column():
            gr.Markdown("## Examples")
            example_inps = [
                [
                    'example_inputs/woman1.png',
                    'ip',
                    'profile shot dark photo of a 25-year-old female with smoke escaping from her mouth, the backlit smoke gives the image an ephemeral quality, natural face, natural eyebrows, natural skin texture, award winning photo, highly detailed face, atmospheric lighting, film grain, monochrome',
                    9180879731249039735,
                ],
                [
                    'example_inputs/man1.png',
                    'ip',
                    'a man sitting on the cloud, playing guitar',
                    1206523688721442817,
                ],
                [
                    'example_inputs/toy1.png',
                    'ip',
                    'a purple toy holding a sign saying "DreamO", on the mountain',
                    10441727852953907380,
                ],
                [
                    'example_inputs/perfume.png',
                    'ip',
                    'a perfume under spotlight',
                    116150031980664704,
                ],
            ]
            gr.Examples(examples=example_inps, inputs=[ref_image1, ref_task1, prompt, seed], label='IP task')

            example_inps = [
                [
                    'example_inputs/hinton.jpeg',
                    'id',
                    'portrait, Chibi',
                    5443415087540486371,
                ],
            ]
            gr.Examples(
                examples=example_inps,
                inputs=[ref_image1, ref_task1, prompt, seed],
                label='ID task (similar to PuLID, will only refer to the face)',
            )

            example_inps = [
                [
                    'example_inputs/mickey.png',
                    'style',
                    'generate a same style image. A rooster wearing overalls.',
                    6245580464677124951,
                ],
                [
                    'example_inputs/mountain.png',
                    'style',
                    'generate a same style image. A pavilion by the river, and the distant mountains are endless',
                    5248066378927500767,
                ],
            ]
            gr.Examples(examples=example_inps, inputs=[ref_image1, ref_task1, prompt, seed], label='Style task')

            example_inps = [
                [
                    'example_inputs/shirt.png',
                    'example_inputs/skirt.jpeg',
                    'ip',
                    'ip',
                    'A girl is wearing a short-sleeved shirt and a short skirt on the beach.',
                    9514069256241143615,
                ],
                [
                    'example_inputs/woman2.png',
                    'example_inputs/dress.png',
                    'id',
                    'ip',
                    'the woman wearing a dress, In the banquet hall',
                    7698454872441022867,
                ],
            ]
            gr.Examples(
                examples=example_inps,
                inputs=[ref_image1, ref_image2, ref_task1, ref_task2, prompt, seed],
                label='Try-On task',
            )

            example_inps = [
                [
                    'example_inputs/dog1.png',
                    'example_inputs/dog2.png',
                    'ip',
                    'ip',
                    'two dogs in the jungle',
                    6187006025405083344,
                ],
                [
                    'example_inputs/woman3.png',
                    'example_inputs/cat.png',
                    'ip',
                    'ip',
                    'A girl rides a giant cat, walking in the noisy modern city. High definition, realistic, non-cartoonish. Excellent photography work, 8k high definition.',
                    11980469406460273604,
                ],
                [
                    'example_inputs/man2.jpeg',
                    'example_inputs/woman4.jpeg',
                    'ip',
                    'ip',
                    'a man is dancing with a woman in the room',
                    8303780338601106219,
                ],
            ]
            gr.Examples(
                examples=example_inps,
                inputs=[ref_image1, ref_image2, ref_task1, ref_task2, prompt, seed],
                label='Multi IP',
            )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                ref_image1,
                ref_image2,
                ref_task1,
                ref_task2,
                prompt,
                width,
                height,
                ref_res,
                num_steps,
                guidance,
                seed,
                true_cfg,
                cfg_start_step,
                cfg_end_step,
                neg_prompt,
                neg_guidance,
                first_step_guidance,
                lora_dir, *lora_dropdowns, *lora_weights,
                face_size,
                upscale_factor,
                crop_ratio,
                num_images
            ],
            outputs=[output_gallery, debug_image, seed_output],
        )

    return demo

if __name__ == '__main__':
    demo = create_demo()
    demo.queue().launch(server_name='127.0.0.1', server_port=args.port)
